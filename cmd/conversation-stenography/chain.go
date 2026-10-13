package main

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	conversationstenography "conversationstenography"
)

// runChain implements the "chain-send", "chain-receive", "chain-show",
// "chain-chat", and "chat" subcommands, which maintain a persistent,
// authenticated multi-party conversation chain.

type persistedChainState struct {
	Version      int                                   `json:"version"`
	Conversation string                                `json:"conversation"`
	Records      []conversationstenography.ChainRecord `json:"records"`
	Decrypted    map[string]string                     `json:"decrypted,omitempty"`
}

const chainStateVersion = 2

func runChain(mode string, args []string, in io.Reader, out, errOut io.Writer) error {
	local, err := loadLocalGenerativeConfig(resolveSupportFile("conversation-stenography.local.json"))
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet(mode, flag.ContinueOnError)
	fs.SetOutput(errOut)
	stateDefault := ".conversation-stenography-chain.json"
	if mode == "chat" {
		stateDefault = ""
	}
	statePath := fs.String("state", stateDefault, "durable local chain state")
	conversation := fs.String("conversation", defaultString(local.Conversation, "default-chat"), "group conversation identifier")
	from := fs.String("from", "", "sender name (chain-send only)")
	me := fs.String("me", "", "your participant name (chat mode)")
	showFormat := fs.String("format", "jsonl", "chain-show format: jsonl or table")
	modelName := fs.String("model", envOr("CONVERSATION_STENOGRAPHY_MODEL", local.Model), "Hugging Face model name or local directory")
	revision := fs.String("revision", envOr("CONVERSATION_STENOGRAPHY_REVISION", defaultString(local.Revision, "main")), "pinned model revision")
	python := fs.String("python", envOr("CONVERSATION_STENOGRAPHY_PYTHON", defaultString(local.Python, "python3")), "Python interpreter")
	runtimeName := fs.String("runtime", envOr("CONVERSATION_STENOGRAPHY_RUNTIME", defaultString(local.Runtime, "transformers")), "transformers or mlx")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("unexpected positional arguments")
	}
	if mode == "chat" {
		if strings.TrimSpace(*me) == "" {
			return errors.New("-me is required for chat")
		}
		*from = *me
		if *statePath == "" {
			*statePath, err = defaultConversationState(*conversation, *me)
			if err != nil {
				return err
			}
		}
	}
	if directory := filepath.Dir(*statePath); directory != "." {
		if err := os.MkdirAll(directory, 0700); err != nil {
			return fmt.Errorf("create state directory: %w", err)
		}
	}
	key, err := conversationKey(*conversation, mode == "chat", errOut)
	if err != nil {
		return err
	}
	state, err := loadChainState(*statePath, *conversation, key)
	if err != nil {
		return err
	}
	if mode == "chain-show" {
		return showChainState(out, state, *showFormat)
	}
	if (mode == "chain-send" || mode == "chain-chat" || mode == "chat") && strings.TrimSpace(*from) == "" {
		return errors.New("-from is required for chain-send and chain-chat")
	}
	if *modelName == "" {
		return errors.New("-model is required (or set CONVERSATION_STENOGRAPHY_MODEL, or run 'conversation-stenography setup')")
	}
	backendPath := resolveSupportFile("python/hf_model.py")
	modelArgs := []string{"--model", *modelName, "--revision", *revision}
	device := envOr("CONVERSATION_STENOGRAPHY_DEVICE", defaultString(local.Device, "cpu"))
	dtype := envOr("CONVERSATION_STENOGRAPHY_DTYPE", defaultString(local.Dtype, "float32"))
	switch *runtimeName {
	case "mlx":
		backendPath = resolveSupportFile("python/mlx_model.py")
	case "transformers":
		modelArgs = append(modelArgs, "--device", device, "--dtype", dtype)
	default:
		return fmt.Errorf("unknown runtime %q", *runtimeName)
	}
	ctx := context.Background()

	if mode == "chat" {
		fmt.Fprintln(out, "  ⏳ Loading model... (this may take a moment)")
	}

	model, err := conversationstenography.NewProcessModel(ctx, *python, append([]string{backendPath}, modelArgs...)...)
	if err != nil {
		return err
	}
	defer model.Close()
	cfg := conversationstenography.GenerativeConfig{Prompt: local.Prompt, TopN: local.TopN, Coding: local.Coding,
		Temperature: local.Temperature, FinishTokens: local.FinishTokens, ChainSystem: local.ChainSystem,
		StrictStyle: local.StrictStyle, CandidatePool: local.CandidatePool, RefreshSentences: local.RefreshSentences,
		CarrierTrials: local.CarrierTrials, NaturalnessSlack: local.NaturalnessSlack, SemanticJudge: local.SemanticJudge,
		SemanticThreshold: local.SemanticThreshold, LengthBias: local.LengthBias}
	chain, err := conversationstenography.NewConversationChain(model, key, state.Conversation, cfg)
	if err != nil {
		return err
	}
	if err := chain.RestorePublic(state.Records); err != nil {
		return fmt.Errorf("restore chain state: %w", err)
	}
	if mode == "chain-chat" || mode == "chat" {
		return interactiveChain(ctx, in, out, errOut, chain, &state, *statePath, *from, mode == "chat", key)
	}
	data, err := io.ReadAll(in)
	if err != nil {
		return err
	}
	if mode == "chain-send" {
		record, err := chain.Send(ctx, *from, data)
		if err != nil {
			return err
		}
		state.Records = chain.Records()
		state.Decrypted[fmt.Sprint(record.Index)] = base64.StdEncoding.EncodeToString(data)
		if err := saveChainState(*statePath, state, key); err != nil {
			return err
		}
		return json.NewEncoder(out).Encode(record)
	}
	var incoming conversationstenography.ChainRecord
	if err := json.Unmarshal(data, &incoming); err != nil {
		return fmt.Errorf("parse chain record: %w", err)
	}
	plaintext, accepted, err := chain.Receive(ctx, incoming.From, incoming.Encrypted)
	if err != nil {
		return err
	}
	if incoming.Index != accepted.Index || incoming.SenderSequence != accepted.SenderSequence {
		return errors.New("chain record metadata does not match expected order")
	}
	state.Records = chain.Records()
	state.Decrypted[fmt.Sprint(accepted.Index)] = base64.StdEncoding.EncodeToString(plaintext)
	if err := saveChainState(*statePath, state, key); err != nil {
		return err
	}
	_, err = out.Write(plaintext)
	return err
}

type encryptedChainState struct {
	Format string `json:"format"`
	Nonce  string `json:"nonce"`
	Body   string `json:"body"`
}

func loadChainState(path, conversation string, key []byte) (persistedChainState, error) {
	state := persistedChainState{Version: chainStateVersion, Conversation: conversation, Decrypted: make(map[string]string)}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return state, nil
	}
	if err != nil {
		return state, fmt.Errorf("read chain state: %w", err)
	}
	var envelope encryptedChainState
	if err := json.Unmarshal(b, &envelope); err == nil && envelope.Format == "DCS1" {
		aead, err := chainStateAEAD(key)
		if err != nil {
			return state, err
		}
		nonce, err := base64.RawStdEncoding.DecodeString(envelope.Nonce)
		if err != nil || len(nonce) != aead.NonceSize() {
			return state, errors.New("invalid encrypted state nonce")
		}
		body, err := base64.RawStdEncoding.DecodeString(envelope.Body)
		if err != nil {
			return state, errors.New("invalid encrypted state body")
		}
		b, err = aead.Open(nil, nonce, body, []byte("decalgo-state-v1\x00"+conversation))
		if err != nil {
			return state, errors.New("cannot decrypt conversation state: wrong phrase, key, or conversation")
		}
	}
	if err := json.Unmarshal(b, &state); err != nil {
		return state, fmt.Errorf("parse conversation state: %w", err)
	}
	if state.Version != chainStateVersion {
		return state, fmt.Errorf("conversation state uses incompatible protocol version %d (current %d); both participants must archive/delete their old state for this conversation and start fresh", state.Version, chainStateVersion)
	}
	if state.Conversation != conversation {
		return state, fmt.Errorf("state conversation %q does not match %q", state.Conversation, conversation)
	}
	if state.Decrypted == nil {
		state.Decrypted = make(map[string]string)
	}
	return state, nil
}

func saveChainState(path string, state persistedChainState, key []byte) error {
	b, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	aead, err := chainStateAEAD(key)
	if err != nil {
		return err
	}
	nonce := make([]byte, aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return err
	}
	body := aead.Seal(nil, nonce, b, []byte("decalgo-state-v1\x00"+state.Conversation))
	envelope, err := json.MarshalIndent(encryptedChainState{Format: "DCS1", Nonce: base64.RawStdEncoding.EncodeToString(nonce), Body: base64.RawStdEncoding.EncodeToString(body)}, "", "  ")
	if err != nil {
		return err
	}
	temporary := path + ".tmp"
	if err := os.WriteFile(temporary, append(envelope, '\n'), 0600); err != nil {
		return fmt.Errorf("write chain state: %w", err)
	}
	if err := os.Rename(temporary, path); err != nil {
		return fmt.Errorf("commit chain state: %w", err)
	}
	return nil
}

func chainStateAEAD(key []byte) (cipher.AEAD, error) {
	material := sha256.New()
	material.Write([]byte("decalgo-state-key-v1\x00"))
	material.Write(key)
	block, err := aes.NewCipher(material.Sum(nil))
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(block)
}

func showChainState(out io.Writer, state persistedChainState, format string) error {
	if format != "jsonl" && format != "table" {
		return errors.New("format must be jsonl or table")
	}
	if format == "table" {
		fmt.Fprintln(out, "from|decrypted|encrypted")
	}
	encoder := json.NewEncoder(out)
	for _, record := range state.Records {
		row := struct {
			From      string `json:"from"`
			Decrypted string `json:"decrypted,omitempty"`
			Encrypted string `json:"encrypted"`
		}{From: record.From, Encrypted: record.Encrypted}
		if encoded := state.Decrypted[fmt.Sprint(record.Index)]; encoded != "" {
			if plaintext, err := base64.StdEncoding.DecodeString(encoded); err == nil {
				row.Decrypted = string(plaintext)
			}
		}
		if format == "table" {
			fmt.Fprintf(out, "%q|%q|%q\n", row.From, row.Decrypted, row.Encrypted)
		} else if err := encoder.Encode(row); err != nil {
			return err
		}
	}
	return nil
}
