package conversationstenography

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

type plainStyleModel struct{ fakeModel }

func (m plainStyleModel) Next(_ context.Context, _ []int, n int) ([]TokenCandidate, error) {
	out := make([]TokenCandidate, n)
	for i := range out {
		out[i] = TokenCandidate{ID: i, LogProb: float64(n - i), Text: " normal"}
	}
	return out, nil
}

type judgeModel struct {
	fakeModel
	approve bool
}

func (m judgeModel) Next(_ context.Context, _ []int, n int) ([]TokenCandidate, error) {
	out := make([]TokenCandidate, n)
	for i := range out {
		out[i] = TokenCandidate{ID: i, LogProb: float64(-i), Text: " other"}
	}
	yes, no := 2.0, 1.0
	if !m.approve {
		yes, no = 1, 2
	}
	out[0] = TokenCandidate{ID: 0, LogProb: yes, Text: " YES"}
	out[1] = TokenCandidate{ID: 1, LogProb: no, Text: " NO"}
	return out, nil
}

func TestModelPromptDoesNotExposeSenderName(t *testing.T) {
	chain := newTestChain(t, "friends")
	chain.baseConfig.ChainSystem = "Write a casual reply."
	chain.records = []ChainRecord{{From: "private-sender-name", Encrypted: "ordinary earlier message"}}
	prompt := chain.messageConfig("another-private-name").Prompt
	if strings.Contains(prompt, "private-sender-name") || strings.Contains(prompt, "another-private-name") {
		t.Fatalf("sender identity leaked into model prompt: %q", prompt)
	}
}

func TestEncodingBudgetContainsOnlyRequiredFields(t *testing.T) {
	chain := newTestChain(t, "budget")
	chain.baseConfig.Coding = "arithmetic"
	message := []byte("i think i might've fucked up a bit too hard")
	budget, err := chain.EncodingBudget(message)
	if err != nil {
		t.Fatal(err)
	}
	if budget.PlaintextBytes != len(message) || budget.AuthenticationBytes != 16 || budget.FrameLengthBytes != 0 || budget.TerminationBytes != 0 {
		t.Fatalf("unexpected budget: %#v", budget)
	}
	if budget.TotalHiddenBytes != budget.PackedBytes+budget.AuthenticationBytes+budget.FrameLengthBytes+budget.TerminationBytes {
		t.Fatalf("budget does not add up: %#v", budget)
	}
	if budget.TotalHiddenBytes >= len(message) {
		t.Fatalf("common message did not shrink after all overhead: %#v", budget)
	}
}

func TestShortChatEncodingBudget(t *testing.T) {
	chain := newTestChain(t, "compact-budget")
	chain.baseConfig.Coding = "arithmetic"
	chain.baseConfig.CarrierTrials = 8
	budget, err := chain.EncodingBudget([]byte("meet me after lunch"))
	if err != nil {
		t.Fatal(err)
	}
	if budget.PackedBytes != 0 || budget.AuthenticationBytes != sivTagSize || budget.FrameLengthBytes != 0 || budget.TerminationBytes != 0 || budget.TotalHiddenBytes != sivTagSize {
		t.Fatalf("unexpected compact budget: %#v", budget)
	}
	if budget.BaselineAuthenticationHypotheses != 8*(len(packingModes())+1)+1 || budget.BaselineAuthenticationBits < 118 {
		t.Fatalf("phrase authentication fell below its bound: %#v", budget)
	}
}

func TestDenseConnectiveEncodingBudget(t *testing.T) {
	chain := newTestChain(t, "dense-budget")
	chain.baseConfig.Coding = "arithmetic"
	message := []byte("i that it is in the and it is for me")
	budget, err := chain.EncodingBudget(message)
	if err != nil {
		t.Fatal(err)
	}
	if budget.PackedBytes != 7 || budget.AuthenticationBytes != 16 || budget.FrameLengthBytes != 0 || budget.TotalHiddenBytes != 23 {
		t.Fatalf("unexpected dense budget: %#v", budget)
	}
}

func newTestChain(t *testing.T, conversation string) *ConversationChain {
	t.Helper()
	chain, err := NewConversationChain(fakeModel{"fixture-v1"}, []byte("0123456789abcdef0123456789abcdef"), conversation,
		GenerativeConfig{Prompt: "P", TopN: 8, Coding: "arithmetic", Temperature: 1})
	if err != nil {
		t.Fatal(err)
	}
	return chain
}

func TestMultiPartyConversationChain(t *testing.T) {
	ctx := context.Background()
	alex := newTestChain(t, "friends")
	sam := newTestChain(t, "friends")

	first, err := sam.Send(ctx, "bob", []byte("hi alex"))
	if err != nil {
		t.Fatal(err)
	}
	got, accepted, err := alex.Receive(ctx, "bob", first.Encrypted)
	if err != nil || string(got) != "hi alex" || accepted.SenderSequence != 0 {
		t.Fatalf("first: %q %#v %v", got, accepted, err)
	}

	reply, err := alex.Send(ctx, "alex", []byte("did you do your homework?"))
	if err != nil {
		t.Fatal(err)
	}
	got, _, err = sam.Receive(ctx, "alex", reply.Encrypted)
	if err != nil || string(got) != "did you do your homework?" {
		t.Fatalf("reply: %q %v", got, err)
	}

	third, err := sam.Send(ctx, "bob", []byte("yes, the maths homework"))
	if err != nil {
		t.Fatal(err)
	}
	got, accepted, err = alex.Receive(ctx, "bob", third.Encrypted)
	if err != nil || string(got) != "yes, the maths homework" || accepted.SenderSequence != 1 || accepted.Index != 2 {
		t.Fatalf("third: %q %#v %v", got, accepted, err)
	}
	if !reflect.DeepEqual(alex.Records(), sam.Records()) {
		t.Fatal("peer public chains diverged")
	}
	karan := newTestChain(t, "friends")
	if err := karan.RestorePublic(alex.Records()); err != nil {
		t.Fatal(err)
	}
	fourth, err := karan.Send(ctx, "karan", []byte("can I join?"))
	if err != nil {
		t.Fatal(err)
	}
	got, accepted, err = alex.Receive(ctx, "karan", fourth.Encrypted)
	if err != nil || string(got) != "can I join?" || accepted.SenderSequence != 0 || accepted.Index != 3 {
		t.Fatalf("new participant: %q %#v %v", got, accepted, err)
	}
}

func TestConversationChainRejectsWrongOrderAndSender(t *testing.T) {
	ctx := context.Background()
	sender := newTestChain(t, "friends")
	first, _ := sender.Send(ctx, "bob", []byte("first"))
	second, _ := sender.Send(ctx, "bob", []byte("second"))

	wrongOrder := newTestChain(t, "friends")
	if _, _, err := wrongOrder.Receive(ctx, "bob", second.Encrypted); err == nil {
		t.Fatal("out-of-order carrier accepted")
	}
	wrongSender := newTestChain(t, "friends")
	if _, _, err := wrongSender.Receive(ctx, "alex", first.Encrypted); err == nil {
		t.Fatal("wrong sender accepted")
	}
	wrongConversation := newTestChain(t, "other")
	if _, _, err := wrongConversation.Receive(ctx, "bob", first.Encrypted); err == nil {
		t.Fatal("wrong conversation accepted")
	}
}

func TestConversationChainRestorePublicState(t *testing.T) {
	ctx := context.Background()
	original := newTestChain(t, "friends")
	first, _ := original.Send(ctx, "bob", []byte("first"))
	restored := newTestChain(t, "friends")
	if err := restored.RestorePublic([]ChainRecord{first}); err != nil {
		t.Fatal(err)
	}
	second, err := restored.Send(ctx, "bob", []byte("second"))
	if err != nil {
		t.Fatal(err)
	}
	if second.Index != 1 || second.SenderSequence != 1 {
		t.Fatalf("bad restored counters: %#v", second)
	}
}

func TestConversationCompressionDictionaryRestoresExactly(t *testing.T) {
	original := newTestChain(t, "dictionary-restore")
	original.records = []ChainRecord{
		{Index: 0, From: "bob", Encrypted: "We should meet beside the northern greenhouse at Riverside Botanical Garden."},
		{Index: 1, From: "alex", Encrypted: "The northern greenhouse works for me."},
	}
	restored := newTestChain(t, "dictionary-restore")
	if err := restored.RestorePublic(original.records); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored.compressionDictionary(), original.compressionDictionary()) {
		t.Fatal("restored conversation derived a different compression dictionary")
	}
	if len(restored.compressionDictionary()) > maxConversationDictionary {
		t.Fatal("conversation dictionary exceeded the DEFLATE window")
	}
}

func TestImplicitCarrierTrialAddsNoPayloadBytes(t *testing.T) {
	chain := newTestChain(t, "trials")
	plaintext := []byte("same message")
	first, err := chain.sealTrial("bob", 0, 0, 0, plaintext)
	if err != nil {
		t.Fatal(err)
	}
	fourth, err := chain.sealTrial("bob", 0, 0, 3, plaintext)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != len(fourth) || reflect.DeepEqual(first, fourth) {
		t.Fatalf("trial must change ciphertext without changing size: %d/%d equal=%v", len(first), len(fourth), reflect.DeepEqual(first, fourth))
	}
	if _, err := chain.openTrials("bob", 0, 0, fourth, 3); err == nil {
		t.Fatal("opened a trial outside the synchronized search range")
	}
	got, err := chain.openTrials("bob", 0, 0, fourth, 4)
	if err != nil || !reflect.DeepEqual(got, plaintext) {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestAuthenticationHypothesisSecurityBound(t *testing.T) {
	hypotheses, err := authenticationHypotheses(1, 8)
	if err != nil {
		t.Fatal(err)
	}
	want := 8*(len(packingModes())+1) + 1
	if hypotheses != want {
		t.Fatalf("got %d hypotheses; want %d", hypotheses, want)
	}
	bits := effectiveAuthenticationBits(hypotheses)
	if bits < 118 || bits >= 128 {
		t.Fatalf("unexpected effective authentication strength %.3f bits", bits)
	}
	maximumCandidates := maxAuthenticationHypotheses / hypotheses
	if _, err := authenticationHypotheses(maximumCandidates, 8); err != nil {
		t.Fatalf("bounded search rejected: %v", err)
	}
	if _, err := authenticationHypotheses(maximumCandidates+1, 8); err == nil {
		t.Fatal("oversized authentication search accepted")
	}
	if floor := effectiveAuthenticationBits(maxAuthenticationHypotheses + hypotheses); floor < 112 {
		t.Fatalf("configured search floor fell below 112 bits: %.3f", floor)
	}
}

func TestPackingModeIsAuthenticatedWithoutPayloadByte(t *testing.T) {
	chain := newTestChain(t, "detached-mode")
	plaintext := []byte("meet me after lunch")
	mode, body, err := packMessageDetached(plaintext, nil)
	if err != nil {
		t.Fatal(err)
	}
	sealed, err := chain.sealTrial("bob", 0, 0, 0, plaintext)
	if err != nil {
		t.Fatal(err)
	}
	if len(body) != 0 || len(sealed) != sivTagSize {
		t.Fatalf("mode consumed payload space: mode=%d body=%d sealed=%d", mode, len(body), len(sealed))
	}
	wrongMode := mode + 1
	if _, err := openSIV(chain.key, chain.trialModeAAD("bob", 0, 0, 0, wrongMode), sealed); err == nil {
		t.Fatal("detached packing mode was not authenticated")
	}
	got, err := chain.openTrials("bob", 0, 0, sealed, 1)
	if err != nil || !reflect.DeepEqual(got, plaintext) {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestInlinePackingModeSIVCompatibility(t *testing.T) {
	chain := newTestChain(t, "inline-mode-compatibility")
	plaintext := []byte("older packed record")
	packed, err := packMessage(plaintext)
	if err != nil {
		t.Fatal(err)
	}
	sealed, err := sealSIV(chain.key, chain.trialAAD("bob", 0, 0, 0), packed)
	if err != nil {
		t.Fatal(err)
	}
	got, err := chain.openTrials("bob", 0, 0, sealed, 1)
	if err != nil || !reflect.DeepEqual(got, plaintext) {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestFramedGenerativeCarrierCompatibility(t *testing.T) {
	ctx := context.Background()
	sender := newTestChain(t, "framed-carrier-compatibility")
	receiver := newTestChain(t, "framed-carrier-compatibility")
	plaintext := []byte("legacy framed carrier")
	sealed, err := sender.sealTrial("bob", 0, 0, 0, plaintext)
	if err != nil {
		t.Fatal(err)
	}
	codec, err := NewGenerativeCodec(sender.model, sender.messageConfig("bob"))
	if err != nil {
		t.Fatal(err)
	}
	carrier, err := codec.Encode(ctx, sealed)
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := receiver.Receive(ctx, "bob", carrier)
	if err != nil || !reflect.DeepEqual(got, plaintext) {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestSendSelectsShortestCarrierTrial(t *testing.T) {
	ctx := context.Background()
	chain := newTestChain(t, "shortest-trial")
	chain.baseConfig.CarrierTrials = 4
	plaintext := []byte("choose the shortest")
	codec, err := NewGenerativeCodec(chain.model, chain.messageConfig("bob"))
	if err != nil {
		t.Fatal(err)
	}
	shortest := 0
	for trial := 0; trial < 4; trial++ {
		sealed, err := chain.sealTrial("bob", 0, 0, trial, plaintext)
		if err != nil {
			t.Fatal(err)
		}
		carrier, _, err := codec.EncodeUnframedWithMetrics(ctx, sealed)
		if err != nil {
			t.Fatal(err)
		}
		length := len([]rune(carrier))
		if shortest == 0 || length < shortest {
			shortest = length
		}
	}
	record, err := chain.Send(ctx, "bob", plaintext)
	if err != nil {
		t.Fatal(err)
	}
	if got := len([]rune(record.Encrypted)); got != shortest {
		t.Fatalf("selected carrier length %d; shortest trial was %d", got, shortest)
	}
}

func TestStrictChainRefusesNonHumanFallback(t *testing.T) {
	chain := newTestChain(t, "strict-carrier")
	chain.model = plainStyleModel{fakeModel{"fixture-v1"}}
	chain.baseConfig.StrictStyle = true
	chain.baseConfig.CandidatePool = 1
	chain.baseConfig.CarrierTrials = 2
	// The strict chain must refuse: neither the acceptance of a non-prose
	// carrier nor the number of trials it burned may be traded away for the
	// clearer wording. The message now reads "usable carrier" because a trial
	// that errors and a trial whose carrier fails the gate are reported together.
	_, err := chain.Send(context.Background(), "bob", []byte("hello"))
	if err == nil {
		t.Fatal("strict chain accepted fake non-prose carrier")
	}
	if !strings.Contains(err.Error(), "usable carrier") {
		t.Fatalf("strict chain failure did not explain that no trial produced a usable carrier: %v", err)
	}
	if !strings.Contains(err.Error(), "sentence-final punctuation") {
		t.Fatalf("strict chain failure did not name the violated rule: %v", err)
	}
}

func TestNaturalnessConstrainedShortestSelection(t *testing.T) {
	options := []carrierOption{
		{text: "too-short", human: true, metrics: CarrierMetrics{VisibleCharacters: 9, MeanLogitRegret: 1}},
		{text: "most natural carrier", human: true, metrics: CarrierMetrics{VisibleCharacters: 20, MeanLogitRegret: 0.2}},
		{text: "balanced choice", human: true, metrics: CarrierMetrics{VisibleCharacters: 15, MeanLogitRegret: 0.4}},
		{text: "bad", human: false, metrics: CarrierMetrics{VisibleCharacters: 3, MeanLogitRegret: 0}},
	}
	if got := selectCarrier(options, true, 0.35); got != "balanced choice" {
		t.Fatalf("selected %q; want shortest carrier inside naturalness band", got)
	}
	if got := selectCarrier(options, false, 0.35); got != "bad" {
		t.Fatalf("non-strict selection got %q; want absolute shortest", got)
	}
}

func TestSemanticMarginInfluencesQualityBand(t *testing.T) {
	options := []carrierOption{
		{text: "short odd", human: true, semanticMargin: -5, metrics: CarrierMetrics{VisibleCharacters: 9, MeanLogitRegret: 0.2}},
		{text: "longer coherent text", human: true, semanticMargin: 2, metrics: CarrierMetrics{VisibleCharacters: 20, MeanLogitRegret: 0.2}},
	}
	if got := selectCarrier(options, true, 0.35); got != "longer coherent text" {
		t.Fatalf("semantic ranking selected %q", got)
	}
}

func TestHumanWrittenCarrierGate(t *testing.T) {
	accepted := []string{
		"I finally watched it last night, and the ending genuinely surprised me.",
		"That sounds pretty good. What did you think?",
	}
	for _, text := range accepted {
		if !humanWrittenCarrier(text) {
			t.Errorf("natural carrier rejected: %q", text)
		}
	}
	rejected := []string{
		"unfinished thought",
		"What? Why? How? Where? When?",
		"This is is obviously broken.",
		"one two three four one two three four.",
		"line one.\nline two.",
	}
	for _, text := range rejected {
		if humanWrittenCarrier(text) {
			t.Errorf("generation failure accepted: %q", text)
		}
	}
}

func TestSemanticHumanWritingJudge(t *testing.T) {
	ctx := context.Background()
	approved, margin, err := semanticHumanWritten(ctx, judgeModel{fakeModel{"judge"}, true}, "I finally made it home, and the rain was ridiculous.")
	if err != nil || !approved || margin <= 0 {
		t.Fatalf("approved prose rejected: approved=%v margin=%f err=%v", approved, margin, err)
	}
	approved, margin, err = semanticHumanWritten(ctx, judgeModel{fakeModel{"judge"}, false}, "One thought. Different topic. Why? Metadata output.")
	if err != nil || approved || margin >= 0 {
		t.Fatalf("bad prose approved: approved=%v margin=%f err=%v", approved, margin, err)
	}
	if _, _, err := semanticHumanWritten(ctx, fakeModel{"missing-answers"}, "ordinary text"); err == nil {
		t.Fatal("judge accepted a model without explicit YES and NO scores")
	}
}
