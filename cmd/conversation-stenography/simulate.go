package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"

	conversationstenography "conversationstenography"
)

// runSimulation creates two independent protocol participants backed by the
// same local model. Every carrier sent by one side is decoded by the other, so
// this exercises the real transport path without requiring another device.
func runSimulation(args []string, in io.Reader, out, errOut io.Writer) error {
	local, err := loadLocalGenerativeConfig(resolveSupportFile("conversation-stenography.local.json"))
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet("simulate", flag.ContinueOnError)
	fs.SetOutput(errOut)
	userA := fs.String("user-a", "Alice", "first simulated participant")
	userB := fs.String("user-b", "Bob", "second simulated participant")
	conversation := fs.String("conversation", "local-simulation", "simulation conversation identifier")
	modelName := fs.String("model", envOr("CONVERSATION_STENOGRAPHY_MODEL", local.Model), "Hugging Face model name or local directory")
	revision := fs.String("revision", envOr("CONVERSATION_STENOGRAPHY_REVISION", defaultString(local.Revision, "main")), "pinned model revision")
	python := fs.String("python", envOr("CONVERSATION_STENOGRAPHY_PYTHON", defaultString(local.Python, "python3")), "Python interpreter")
	runtimeName := fs.String("runtime", envOr("CONVERSATION_STENOGRAPHY_RUNTIME", defaultString(local.Runtime, "transformers")), "transformers or mlx")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("simulate does not accept positional arguments")
	}
	*userA = strings.TrimSpace(*userA)
	*userB = strings.TrimSpace(*userB)
	if *userA == "" || *userB == "" {
		return errors.New("simulated user names cannot be empty")
	}
	if *userA == *userB {
		return errors.New("-user-a and -user-b must be different")
	}
	if strings.TrimSpace(*conversation) == "" {
		return errors.New("-conversation cannot be empty")
	}
	if *modelName == "" {
		return errors.New("-model is required (run 'conversation-stenography setup' first)")
	}

	key, err := conversationKey(*conversation, true, errOut)
	if err != nil {
		return err
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

	fmt.Fprintln(out, "  ⏳ Loading model... (this may take a moment)")
	ctx := context.Background()
	model, err := conversationstenography.NewProcessModel(ctx, *python, append([]string{backendPath}, modelArgs...)...)
	if err != nil {
		return err
	}
	defer model.Close()

	cfg := conversationstenography.GenerativeConfig{
		Prompt: local.Prompt, TopN: local.TopN, Coding: local.Coding,
		Temperature: local.Temperature, FinishTokens: local.FinishTokens, ChainSystem: local.ChainSystem,
		StrictStyle: local.StrictStyle, CandidatePool: local.CandidatePool, RefreshSentences: local.RefreshSentences,
		CarrierTrials: local.CarrierTrials, NaturalnessSlack: local.NaturalnessSlack, SemanticJudge: local.SemanticJudge,
		SemanticThreshold: local.SemanticThreshold, LengthBias: local.LengthBias,
	}
	first, err := conversationstenography.NewConversationChain(model, key, *conversation, cfg)
	if err != nil {
		return err
	}
	second, err := conversationstenography.NewConversationChain(model, key, *conversation, cfg)
	if err != nil {
		return err
	}
	return simulateConversation(ctx, in, out, errOut, first, second, *userA, *userB)
}

func simulateConversation(ctx context.Context, in io.Reader, out, errOut io.Writer, first, second *conversationstenography.ConversationChain, userA, userB string) error {
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 4096), 16*1024*1024)
	activeName, otherName := userA, userB
	activeChain, otherChain := first, second

	fmt.Fprintln(out)
	fmt.Fprintln(out, "  ┌──────────────────────────────────────────┐")
	fmt.Fprintln(out, "  │       🧪  Two-user local simulation      │")
	fmt.Fprintln(out, "  └──────────────────────────────────────────┘")
	fmt.Fprintf(out, "  %s and %s are independent simulated participants.\n", userA, userB)
	fmt.Fprintln(out, "  Type a secret message; the other user will decode it.")
	fmt.Fprintln(out, "  Turns alternate automatically. Use /switch, /show, or /quit.")
	fmt.Fprintln(out)

	var history []string
	for {
		fmt.Fprintf(out, "%s> ", activeName)
		if !scanner.Scan() {
			return scanner.Err()
		}
		plaintext := scanner.Text()
		switch plaintext {
		case "/quit", "/exit":
			fmt.Fprintln(out, "  Simulation ended; no conversation state was saved.")
			return nil
		case "/switch":
			activeName, otherName = otherName, activeName
			activeChain, otherChain = otherChain, activeChain
			continue
		case "/show":
			if len(history) == 0 {
				fmt.Fprintln(out, "  No simulated messages yet.")
			} else {
				for _, line := range history {
					fmt.Fprintln(out, "  "+line)
				}
			}
			continue
		case "/help":
			fmt.Fprintln(out, "  Type a message to send it; /switch changes speaker; /show displays plaintext history; /quit exits.")
			continue
		case "":
			fmt.Fprintln(errOut, "  Message cannot be empty.")
			continue
		}
		if strings.HasPrefix(plaintext, "/") {
			fmt.Fprintln(errOut, "  Unknown command. Type /help for commands.")
			continue
		}

		fmt.Fprintln(out, "  Generating and transporting cover text...")
		record, err := activeChain.Send(ctx, activeName, []byte(plaintext))
		if err != nil {
			return fmt.Errorf("%s send: %w", activeName, err)
		}
		decoded, accepted, err := otherChain.Receive(ctx, activeName, record.Encrypted)
		if err != nil {
			return fmt.Errorf("%s receive: %w", otherName, err)
		}
		if accepted.Index != record.Index || accepted.SenderSequence != record.SenderSequence {
			return errors.New("simulation participants produced mismatched chain metadata")
		}

		fmt.Fprintln(out)
		fmt.Fprintln(out, "  Cover text (what the messaging app would see):")
		fmt.Fprintf(out, "  %s\n\n", record.Encrypted)
		fmt.Fprintf(out, "  %s decoded: %s\n\n", otherName, decoded)
		history = append(history, fmt.Sprintf("%s → %s: %s", activeName, otherName, plaintext))
		activeName, otherName = otherName, activeName
		activeChain, otherChain = otherChain, activeChain
	}
}
