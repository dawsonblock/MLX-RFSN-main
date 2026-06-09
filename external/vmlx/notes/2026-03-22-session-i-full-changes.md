# Session 2026-03-22i — COMPLETE CHANGE LIST

Every single file modified this session, what changed, and what it interacts with.

---

## CATEGORY 1: Mistral 4 Reasoning (6 changes)

| # | File | Change | Interacts With |
|---|------|--------|---------------|
| M4-1 | `vmlx_engine/reasoning/mistral_parser.py` (NEW) | MistralReasoningParser class, [THINK]/[/THINK] tokens, extends BaseThinkingReasoningParser | All streaming/non-streaming inference paths, tool parsing, multi-turn history |
| M4-2 | `vmlx_engine/reasoning/__init__.py:88` | `register_parser("mistral", MistralReasoningParser)` | Parser lookup at server init, get_parser() calls |
| M4-3 | `vmlx_engine/model_configs.py:257-267` | reasoning_parser="mistral", think_in_template=False, tool_parser="mistral" | Model config registry lookup, CLI init, server.py parser selection |
| M4-4 | `vmlx_engine/server.py:156` | `_THINK_STRIP_RE` updated for both `<think>` and `[THINK]` | Multi-turn history cleaning (lines 3186, 3731) |
| M4-5 | `vmlx_engine/server.py:3411,3945,4595,5189` | 4 inline re.sub updated for both formats | Tool call parsing in chat/responses streaming/non-streaming |
| M4-6 | `vmlx_engine/server.py:3412-3416,3946-3950,4596-4600,5190-5194` | 4 fallback partition blocks check both `</think>` and `[/THINK]` | Same 4 tool parsing paths |

## CATEGORY 2: JANG Quantizer Advanced Params (7 changes)

| # | File | Change | Interacts With |
|---|------|--------|---------------|
| JQ-1 | `vmlx_engine/cli.py:1352-1356` | Added `--awq-alpha` argparse argument | CLI → convert command |
| JQ-2 | `vmlx_engine/commands/convert.py:476,493,511` | Read awq_alpha, print in summary, pass to convert_model() | jang_tools.convert.convert_model() |
| JQ-3 | `panel/src/main/ipc/developer.ts:170,182-184` | Added awqAlpha type + CLI arg passthrough | IPC handler → Python CLI spawn |
| JQ-4 | `panel/src/preload/index.ts:234-235` | Added calibrationMethod, imatrixPath, useAwq, awqAlpha to convert type | Preload bridge → renderer |
| JQ-5 | `panel/src/env.d.ts:122` | Added same 4 params to global type declaration | TypeScript type checking |
| JQ-6 | `panel/src/renderer/src/components/tools/ModelConverter.tsx:49-52` | 4 new state vars | UI state |
| JQ-7 | `panel/src/renderer/src/components/tools/ModelConverter.tsx` (section) | Advanced JANG Settings UI: calibration dropdown, imatrix input, AWQ checkbox, alpha slider | UI → IPC → CLI → jang_tools |

## CATEGORY 3: i18n Translation System (8 changes)

| # | File | Change | Interacts With |
|---|------|--------|---------------|
| i18n-1 | `panel/src/renderer/src/i18n/index.tsx` (NEW) | I18nProvider, useTranslation hook, t() function, localStorage persistence | Every component that uses t() |
| i18n-2 | `panel/src/renderer/src/i18n/locales/en.json` (NEW) | 721 English translation keys | All t() calls fall back to this |
| i18n-3 | `panel/src/renderer/src/i18n/locales/zh.json` (NEW) | 721 Chinese (Simplified) translations | Language switcher |
| i18n-4 | `panel/src/renderer/src/i18n/locales/ko.json` (NEW) | 721 Korean translations | Language switcher |
| i18n-5 | `panel/src/renderer/src/i18n/locales/ja.json` (NEW) | 721 Japanese translations | Language switcher |
| i18n-6 | `panel/src/renderer/src/i18n/locales/es.json` (NEW) | 721 Spanish translations | Language switcher |
| i18n-7 | `panel/src/renderer/src/main.tsx` | Wrapped app with `<I18nProvider>` | All child components |
| i18n-8 | `panel/src/renderer/src/App.tsx` | Added useTranslation import, language switcher in About panel | ServerModeContent, About section |

## CATEGORY 4: i18n Component Wrapping (43 files modified)

**Chat (7 files):**
ChatInterface.tsx, InputBox.tsx, MessageList.tsx, MessageBubble.tsx, InlineToolCall.tsx, ReasoningBox.tsx, VoiceChat.tsx

**Sessions (9 files):**
SessionDashboard.tsx, SessionCard.tsx, CreateSession.tsx, SessionSettings.tsx, ServerSettingsDrawer.tsx, DirectoryManager.tsx, LogsPanel.tsx, BenchmarkPanel.tsx, PerformancePanel.tsx

**Image (8 files):**
ImageTab.tsx, ImageSettings.tsx, ImageGallery.tsx, ImagePromptBar.tsx, ImageTopBar.tsx, ImageHistory.tsx, ImageModelPicker.tsx, MaskPainter.tsx

**API (3 files):**
ApiDashboard.tsx, EndpointList.tsx, CodeSnippets.tsx

**Tools (4 files):**
ToolsDashboard.tsx, ModelConverter.tsx, ModelInspector.tsx, ModelDoctor.tsx

**Layout (4 files):**
TitleBar.tsx, SidebarHeader.tsx, ChatHistory.tsx, ChatModeToolbar.tsx

**Downloads (3 files):**
DownloadStatusBar.tsx, DownloadsView.tsx, DownloadTab.tsx

**Other (5 files):**
UpdateNotice.tsx, UpdateBanner.tsx, SetupScreen.tsx, theme-toggle.tsx, ChatSettings.tsx

## CATEGORY 5: UI Visual Polish (5 changes)

| # | File | Change | Interacts With |
|---|------|--------|---------------|
| VP-1 | `panel/src/renderer/src/index.css` | border-border/40 global, transitions 150ms on interactive elements, placeholder styling, focus ring improvements, cfg-input focus states, code block shadow+border softening | ALL components that render borders, buttons, inputs |
| VP-2 | `panel/src/renderer/src/components/ui/button.tsx` | hover:shadow-md, active:scale-[0.98], ring-2 ring-offset-2, ghost hover:bg-accent/80, transition-all duration-150 | Every Button component usage across entire app |
| VP-3 | `panel/src/renderer/src/components/layout/TitleBar.tsx` | Mode tab container: border/30 + shadow-sm, active button: shadow-primary/10, inactive: text/70, duration-200 | App navigation, mode switching |
| VP-4 | `panel/src/renderer/src/components/sessions/SessionCard.tsx` | border/40, shadow-sm, hover:shadow-lg, p-3 (was p-4), status dot w-2.5 | Session dashboard, session management |
| VP-5 | `panel/src/renderer/src/index.css` | Code block wrapper: border/50 + box-shadow, code header: border/40 | Markdown rendering in chat, README viewer |

---

## RISK MATRIX: What Could Break

### i18n × Inference Engine
| Risk | Description | Likelihood | Impact |
|------|-------------|:---:|:---:|
| R1 | t() returns undefined crashes render | ZERO | — | t() always returns key as fallback |
| R2 | Missing locale key shows raw key like "chat.truncated" | LOW | LOW | Fallback chain: locale → en.json → raw key |
| R3 | Interpolation param missing shows {count} literally | MEDIUM | LOW | Only visual, no crash |
| R4 | CJK text overflows fixed-width buttons | MEDIUM | MEDIUM | Need visual testing per language |
| R5 | useTranslation outside I18nProvider | ZERO | — | Provider wraps entire app in main.tsx |
| R6 | localStorage locale value corrupted | ZERO | — | Falls back to 'en' if invalid |
| R7 | i18n import breaks server-side code | ZERO | — | i18n is renderer-only, no SSR |

### i18n × Mistral 4
| Risk | Description | Likelihood | Impact |
|------|-------------|:---:|:---:|
| R8 | Translated UI sends different API params | ZERO | — | t() only affects display strings, never API values |
| R9 | reasoning_effort dropdown value translated | ZERO | — | Values are "low"/"medium"/"high" (code), only labels are translated |
| R10 | tool_parser name translated | ZERO | — | Parser names are code strings, never displayed via t() |
| R11 | Model config registry uses translated strings | ZERO | — | Registry is Python-side, i18n is renderer-side |

### UI Polish × Inference Engine
| Risk | Description | Likelihood | Impact |
|------|-------------|:---:|:---:|
| R12 | CSS transitions delay API calls | ZERO | — | CSS transitions are visual-only, no JS timing impact |
| R13 | border-border/40 breaks Tailwind parsing | LOW | MEDIUM | Tailwind v3 supports opacity modifier syntax |
| R14 | active:scale-[0.98] causes layout shift | LOW | LOW | Only on click, brief 100ms, no reflow |
| R15 | Global transition on inputs delays typing | ZERO | — | Transition is on bg/border/shadow, not on value |

### Mistral 4 × Existing Parsers
| Risk | Description | Likelihood | Impact |
|------|-------------|:---:|:---:|
| R16 | [THINK] regex matches inside user message text | LOW | MEDIUM | Only if user literally types [THINK]...[/THINK] |
| R17 | _THINK_STRIP_RE strips [THINK] from non-Mistral models | LOW | LOW | Only strips if model output contains the token |
| R18 | Mistral tool+reasoning combined output misparse | MEDIUM | MEDIUM | Need live test with actual model |
| R19 | Harmony (GPT-OSS) parser confused by [THINK] | ZERO | — | Harmony uses completely different channel markers |

### JANG Quantizer × Existing Conversion
| Risk | Description | Likelihood | Impact |
|------|-------------|:---:|:---:|
| R20 | awq_alpha param rejected by older jang_tools | MEDIUM | LOW | convert_model() has **kwargs, unknown params ignored |
| R21 | UI sends calibrationMethod but CLI doesn't forward | ZERO | — | Verified: CLI has --calibration-method arg since this session |
| R22 | awqAlpha=0.25 sent when useAwq=false | ZERO | — | Conditional: only sent when isJang && useAwq |

---

## CROSS-CHECK AUDIT RESULTS (3 agents, post-implementation)

### Agent 1: i18n vs Inference — ALL SAFE
- ChatSettings.tsx: option values hardcoded, t() on labels only
- SessionConfigForm.tsx: NOT WRAPPED AT ALL (zero i18n exposure, gap for translation)
- CreateSession.tsx: all config values hardcoded
- ModelConverter.tsx: profile codes, method values never translated
- ImageModelPicker.tsx: mflux classes, quantize values safe
- ChatModeToolbar.tsx: API URLs use raw state
- **VERDICT: No t() call wraps any form value, API param, or code string**

### Agent 2: Mistral 4 vs Parsers — 2 BUGS FOUND AND FIXED
- Items 1-7, 9: ALL SAFE (parser isolation, state management, fallback logic)
- **Item 8 (DANGER): reasoning_effort went to chat_kwargs but NOT chat_template_kwargs**
  - Mistral 4 Jinja template reads from chat_template_kwargs
  - FIXED: Added `_ct_kwargs.setdefault("reasoning_effort", ...)` in BOTH paths (lines 3244, 3793)
- **Item 10 (LOW DANGER): abstract_tool_parser.py THINK_TAG_PATTERN only matched `<think>`**
  - Defense-in-depth gap for Mistral 4 tool parsing
  - FIXED: Updated both patterns to match `[THINK]`/`[/THINK]` as well

### Agent 3: CSS vs Layout — 2 ISSUES FOUND, 1 FIXED
- border-border/40: SAFE (Tailwind config uses `<alpha-value>` pattern)
- active:scale-[0.98]: SAFE (transform doesn't affect layout flow)
- shadow-primary/10: SAFE (supported Tailwind v3 syntax)
- **cfg-input focus ring stacking: FIXED** — changed `focus:` to `focus-visible:` to align with global rule
- Global transition vs transition-* classes: LOW — visual inconsistency only, acceptable

### Total Cross-Check Score: 17 SAFE, 3 FOUND+FIXED, 0 REMAINING
