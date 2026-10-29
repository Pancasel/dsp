"""CSS of the whole page (soft green theme, cards, chips, animations)."""

# Class names used by the Python code:
# effect-card-head / -rank / -name / -english / -pairs: one side-effect card
# tier-badge: coloured Tier label, drug-chip: a drug in the regimen,
# answer-summary: the line above the cards of an answer.
PAGE_CSS = """
<style>
:root {
  --ink: #17352f;
  --muted: #668078;
  --line: rgba(103, 151, 135, .17);
  --mint: #14977d;
}
.stApp {
  background:
    radial-gradient(ellipse at 4% 8%, rgba(92, 205, 174, .14), transparent 31rem),
    radial-gradient(ellipse at 96% 43%, rgba(122, 181, 232, .12), transparent 34rem),
    linear-gradient(145deg, #f5fbf8 0%, #f7fafc 50%, #f2f8f7 100%);
  color: var(--ink);
}
.stApp::before, .stApp::after {
  content: "";
  position: fixed;
  z-index: 0;
  pointer-events: none;
  width: 25rem;
  height: 25rem;
  border-radius: 50%;
  filter: blur(72px);
  opacity: .18;
  animation: drift 22s ease-in-out infinite alternate;
}
.stApp::before {background: #72d7b4; top: 8%; left: -17rem;}
.stApp::after {background: #9ac9ee; right: -17rem; bottom: 3%; animation-delay: -9s;}
@keyframes drift {
  from {transform: translate3d(0, -12px, 0) scale(.94);}
  to {transform: translate3d(32px, 30px, 0) scale(1.08);}
}
.block-container {
  position: relative;
  z-index: 1;
  max-width: 1040px;
  padding-top: 2rem;
  padding-bottom: 3rem;
}
h1, h2, h3, [data-testid="stMetricValue"] {
  color: var(--ink) !important;
  font-family: 'Aptos Display', 'Segoe UI', sans-serif;
  letter-spacing: -.035em;
}
h1 {font-size: 2rem !important; margin-bottom: .35rem;}
p, label, li {font-family: 'Aptos', 'Segoe UI', sans-serif;}
[data-testid="stCaptionContainer"] {color: var(--muted);}
[data-testid="stSidebar"] {
  background: rgba(246, 251, 248, .88);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child {background: transparent;}
[data-testid="stChatMessage"] {
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1rem 1.15rem;
  background: rgba(255, 255, 255, .76);
  box-shadow: 0 8px 28px rgba(34, 83, 68, .045);
  animation: rise-in .35s ease-out both;
  backdrop-filter: blur(10px);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: linear-gradient(135deg, rgba(229, 247, 240, .96), rgba(238, 247, 251, .94));
  border-color: rgba(72, 155, 128, .2);
}
@keyframes rise-in {
  from {opacity: 0; transform: translateY(8px);}
  to {opacity: 1; transform: translateY(0);}
}
[data-testid="stChatInput"] textarea {
  border: 1px solid rgba(67, 137, 114, .25);
  border-radius: 18px;
  background: rgba(255, 255, 255, .9);
  box-shadow: 0 8px 28px rgba(34, 83, 68, .07);
}
[data-testid="stChatInput"] textarea:focus {
  border-color: rgba(20, 151, 125, .72);
  box-shadow: 0 0 0 3px rgba(20, 151, 125, .12);
}
div.stButton > button, [data-testid="stDownloadButton"] button {
  border-radius: 12px;
  border: 1px solid rgba(61, 138, 115, .18);
  color: #24574a;
  background: rgba(255, 255, 255, .78);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
div.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
  transform: translateY(-2px);
  border-color: rgba(20, 151, 125, .48);
  box-shadow: 0 8px 18px rgba(34, 83, 68, .1);
  color: #116b59;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--line) !important;
  border-radius: 18px !important;
  background: rgba(255, 255, 255, .76);
  box-shadow: 0 8px 28px rgba(34, 83, 68, .045);
  transition: transform .2s ease, box-shadow .2s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 13px 32px rgba(34, 83, 68, .09);
}
[data-testid="stExpander"] {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255, 255, 255, .58);
}
[data-testid="stTabs"] [role="tab"] {border-radius: 10px 10px 0 0;}
[data-testid="stAlert"] {border-radius: 14px;}
.effect-card-head {display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 4px;}
.effect-card-rank {
  background: linear-gradient(145deg, #e7f4ff, #e5f7ef);
  color: #256552;
  font-weight: 700;
  border: 1px solid rgba(86, 157, 132, .16);
  border-radius: 50%;
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85rem;
  flex: none;
}
.effect-card-name {font-size: 1.05rem; font-weight: 700; color: var(--ink);}
.effect-card-english {color: var(--muted); font-size: 0.85rem;}
.tier-badge {color: #fff; border-radius: 999px; padding: 3px 11px; font-size: 0.75rem; font-weight: 650;}
.effect-card-pairs {color: var(--muted); font-size: 0.85rem; margin: 5px 0 4px;}
.drug-chip {
  display: inline-block;
  color: #24574a;
  background: linear-gradient(135deg, #e8f6ef, #eef6fb);
  border: 1px solid rgba(86, 157, 132, .14);
  border-radius: 999px;
  padding: 4px 12px;
  margin: 2px 4px 2px 0;
  font-size: 0.9rem;
}
.answer-summary {
  color: #45675d;
  font-size: 0.92rem;
  margin-bottom: .8rem;
  padding: .75rem 1rem;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: linear-gradient(110deg, rgba(230, 247, 239, .78), rgba(239, 247, 251, .76));
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
  }
}
.st-key-chat-view, .st-key-regimen-view {
  animation: page-enter .3s cubic-bezier(.2, .75, .25, 1) both;
}
@keyframes page-enter {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
}
[data-testid="stRadio"] [role="radiogroup"] {gap: .5rem;}
[data-testid="stRadio"] label {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: .45rem .8rem;
  background: rgba(255, 255, 255, .58);
}
[data-testid="stRadio"] label:has(input:checked) {
  border-color: rgba(20, 151, 125, .35);
  background: linear-gradient(110deg, rgba(226, 246, 237, .95), rgba(234, 245, 251, .92));
}
</style>
"""
