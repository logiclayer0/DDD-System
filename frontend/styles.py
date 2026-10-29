"""Shared responsive styling for the Streamlit application."""

APP_CSS = """
<style>
:root {
  --navy: #0A192F;
  --cyan: #64FFDA;
  --white: #FFFFFF;
  --light-gray: #F3F6FA;
  --muted: #627086;
}
.stApp { background: #F3F6FA; color: var(--navy); }
[data-testid="stSidebar"] { background: var(--navy); }
[data-testid="stSidebar"] * { color: var(--white); }
.metric-card, .info-card {
  background: var(--white); border: 1px solid #E1E7EF; border-radius: 14px;
  padding: 18px 20px; margin: 6px 0; box-shadow: 0 5px 18px rgba(10,25,47,.06);
  min-height: 100px;
}
.metric-label { color: var(--muted); font-size: .82rem; font-weight: 600; }
.metric-value { color: var(--navy); font-size: 1.65rem; font-weight: 750; margin-top: 6px; }
.metric-detail { color: var(--muted); font-size: .78rem; margin-top: 3px; }
.info-card h3 { color: var(--navy); font-size: 1.08rem; margin: 0 0 8px; }
.info-card p, .section-header p { color: var(--muted); margin: 0; }
.section-header { border-left: 4px solid var(--cyan); padding-left: 14px; margin: 18px 0; }
.section-header h2 { color: var(--navy); margin: 0 0 4px; font-size: 1.55rem; }
.status-banner {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  border-radius: 12px; padding: 14px 18px; margin: 12px 0;
  background: #E7EDF3; color: var(--navy);
}
.status-success { background: #E4F7EF; border-left: 5px solid #25A36F; }
.status-warning { background: #FFF4D6; border-left: 5px solid #E1A800; }
.status-danger { background: #FCE8E8; border-left: 5px solid #D64545; }
.status-neutral { background: #E7EDF3; border-left: 5px solid #718096; }
.hero {
  background: linear-gradient(125deg, #0A192F, #153C55); color: white; padding: 32px;
  border-radius: 18px; margin-bottom: 20px; box-shadow: 0 10px 24px rgba(10,25,47,.12);
}
.hero h1 { color: white; margin: 0 0 8px; }
.hero p { color: #D7E4EF; max-width: 760px; margin: 0; }
@media (max-width: 700px) {
  .metric-card, .info-card { padding: 14px; min-height: 84px; }
  .metric-value { font-size: 1.3rem; }
  .hero { padding: 22px; }
  .status-banner { align-items: flex-start; flex-direction: column; }
}
</style>
"""
