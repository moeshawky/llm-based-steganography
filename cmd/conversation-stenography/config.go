package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// localGenerativeConfig mirrors conversation-stenography.local.json, the optional local
// defaults file for the generative and conversation-chain subcommands. Device and dtype
// select the transformers runtime's compute target (e.g. "cuda" + "float32"); empty
// values keep the portable cpu/float32 defaults.
type localGenerativeConfig struct {
	Runtime           string  `json:"runtime"`
	Python            string  `json:"python"`
	Model             string  `json:"model"`
	Revision          string  `json:"revision"`
	Device            string  `json:"device"`
	Dtype             string  `json:"dtype"`
	Prompt            string  `json:"prompt"`
	TopN              int     `json:"top_n"`
	Coding            string  `json:"coding"`
	Temperature       float64 `json:"temperature"`
	Secure            bool    `json:"secure"`
	Conversation      string  `json:"conversation"`
	Direction         string  `json:"direction"`
	FinishTokens      int     `json:"finish_tokens"`
	ChainSystem       string  `json:"chain_system"`
	StrictStyle       bool    `json:"strict_style"`
	CandidatePool     int     `json:"candidate_pool"`
	RefreshSentences  bool    `json:"refresh_sentences"`
	CarrierTrials     int     `json:"carrier_trials"`
	NaturalnessSlack  float64 `json:"naturalness_slack"`
	SemanticJudge     bool    `json:"semantic_judge"`
	SemanticThreshold float64 `json:"semantic_threshold"`
	LengthBias        float64 `json:"length_bias"`
}

func loadLocalGenerativeConfig(path string) (localGenerativeConfig, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return localGenerativeConfig{}, nil
	}
	if err != nil {
		return localGenerativeConfig{}, fmt.Errorf("read %s: %w", path, err)
	}
	var cfg localGenerativeConfig
	if err := json.Unmarshal(b, &cfg); err != nil {
		return cfg, fmt.Errorf("parse %s: %w", path, err)
	}
	return cfg, nil
}

func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	// Accept the former product's variables so existing installations continue
	// to work during the rename.
	if strings.HasPrefix(name, "CONVERSATION_STENOGRAPHY_") {
		legacy := "DECALGO_" + strings.TrimPrefix(name, "CONVERSATION_STENOGRAPHY_")
		if value := os.Getenv(legacy); value != "" {
			return value
		}
	}
	return fallback
}

func defaultString(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}

func resolveSupportFile(name string) string {
	if name == "conversation-stenography.local.json" {
		if configured := envOr("CONVERSATION_STENOGRAPHY_CONFIG", ""); configured != "" {
			return configured
		}
		// Prefer the renamed config but transparently reuse an existing config
		// from before the product rename.
		if _, err := os.Stat(name); errors.Is(err, os.ErrNotExist) {
			if _, legacyErr := os.Stat("decalgo.local.json"); legacyErr == nil {
				return "decalgo.local.json"
			}
		}
	}
	if _, err := os.Stat(name); err == nil {
		return name
	}
	if executable, err := os.Executable(); err == nil {
		candidate := filepath.Join(filepath.Dir(executable), name)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		if name == "conversation-stenography.local.json" {
			legacy := filepath.Join(filepath.Dir(executable), "decalgo.local.json")
			if _, err := os.Stat(legacy); err == nil {
				return legacy
			}
		}
	}
	return name
}
