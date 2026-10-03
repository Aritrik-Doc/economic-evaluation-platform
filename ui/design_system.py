"""Shared visual language for the Economic Evaluation Platform."""

from __future__ import annotations

import html
from typing import Iterable

import streamlit as st


PALETTE = {
    "navy": "#16324F",
    "teal": "#2A7F8E",
    "blue": "#4776A6",
    "page": "#F6F8FA",
    "card": "#FFFFFF",
    "teal_soft": "#EAF5F5",
    "blue_soft": "#EAF1F8",
    "green": "#3F7D5A",
    "green_soft": "#EAF4ED",
    "amber": "#D99A32",
    "amber_soft": "#FFF6E5",
    "red": "#B85450",
    "red_soft": "#FBECEC",
    "text": "#263238",
    "muted": "#60717A",
    "border": "#DCE4E8",
}


def apply_design_system() -> None:
    """Inject a restrained scientific/HTA visual theme into the current page."""
    st.markdown(
        f"""
<style>
:root {{
  --eep-navy: {PALETTE['navy']};
  --eep-teal: {PALETTE['teal']};
  --eep-blue: {PALETTE['blue']};
  --eep-page: {PALETTE['page']};
  --eep-card: {PALETTE['card']};
  --eep-text: {PALETTE['text']};
  --eep-muted: {PALETTE['muted']};
  --eep-border: {PALETTE['border']};
}}

[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, #FBFCFD 0%, {PALETTE['page']} 100%);
  color: var(--eep-text);
}}

[data-testid="stHeader"] {{
  background: rgba(246, 248, 250, 0.82);
  backdrop-filter: blur(8px);
}}

[data-testid="stSidebar"] {{
  background: #F1F5F7;
  border-right: 1px solid {PALETTE['border']};
}}

.block-container {{
  max-width: 1280px;
  padding-top: 2rem;
  padding-bottom: 4rem;
}}

h1, h2, h3 {{
  color: var(--eep-navy);
  letter-spacing: -0.018em;
}}

p, li, label {{
  color: var(--eep-text);
}}

.eep-hero {{
  background: linear-gradient(135deg, {PALETTE['navy']} 0%, #204F68 58%, {PALETTE['teal']} 100%);
  color: white;
  border-radius: 22px;
  padding: 2.2rem 2.4rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 14px 36px rgba(22, 50, 79, 0.15);
}}
.eep-hero .eyebrow {{
  color: #BEE5E7;
  font-size: .79rem;
  letter-spacing: .13em;
  font-weight: 700;
  text-transform: uppercase;
  margin-bottom: .5rem;
}}
.eep-hero h1 {{
  color: white;
  margin: 0 0 .8rem 0;
  font-size: clamp(2rem, 5vw, 3.2rem);
  line-height: 1.08;
}}
.eep-hero p {{
  color: #EEF6F7;
  max-width: 880px;
  font-size: 1.06rem;
  line-height: 1.7;
  margin: 0;
}}

.eep-card {{
  background: {PALETTE['card']};
  border: 1px solid {PALETTE['border']};
  border-radius: 16px;
  padding: 1.15rem 1.25rem;
  height: 100%;
  box-shadow: 0 6px 18px rgba(22, 50, 79, 0.045);
}}
.eep-card h3 {{
  margin-top: 0;
  margin-bottom: .42rem;
  font-size: 1.04rem;
}}
.eep-card p {{
  color: {PALETTE['muted']};
  line-height: 1.55;
  margin-bottom: 0;
}}

.eep-block {{
  border-radius: 16px;
  padding: 1.2rem 1.35rem;
  border: 1px solid {PALETTE['border']};
  margin: .5rem 0 1rem 0;
}}
.eep-block-blue {{ background: {PALETTE['blue_soft']}; }}
.eep-block-teal {{ background: {PALETTE['teal_soft']}; }}
.eep-block-green {{ background: {PALETTE['green_soft']}; }}
.eep-block-amber {{ background: {PALETTE['amber_soft']}; }}
.eep-block-red {{ background: {PALETTE['red_soft']}; }}

.eep-kicker {{
  color: {PALETTE['teal']};
  text-transform: uppercase;
  font-size: .75rem;
  font-weight: 750;
  letter-spacing: .12em;
  margin-bottom: .25rem;
}}

.eep-step {{
  background: white;
  border-left: 4px solid {PALETTE['teal']};
  border-radius: 10px;
  padding: .85rem 1rem;
  margin: .45rem 0;
  border-top: 1px solid {PALETTE['border']};
  border-right: 1px solid {PALETTE['border']};
  border-bottom: 1px solid {PALETTE['border']};
}}
.eep-step strong {{ color: {PALETTE['navy']}; }}
.eep-step span {{ color: {PALETTE['muted']}; }}

.eep-badge {{
  display: inline-block;
  border-radius: 999px;
  padding: .24rem .62rem;
  font-size: .76rem;
  font-weight: 700;
  margin: .12rem .22rem .12rem 0;
}}
.eep-badge-green {{ background: {PALETTE['green_soft']}; color: {PALETTE['green']}; }}
.eep-badge-amber {{ background: {PALETTE['amber_soft']}; color: #8B611C; }}
.eep-badge-blue {{ background: {PALETTE['blue_soft']}; color: {PALETTE['blue']}; }}
.eep-badge-teal {{ background: {PALETTE['teal_soft']}; color: {PALETTE['teal']}; }}
.eep-badge-red {{ background: {PALETTE['red_soft']}; color: {PALETTE['red']}; }}
.eep-badge-neutral {{ background: #EEF2F4; color: {PALETTE['muted']}; }}

.eep-statusbar {{
  background: white;
  border: 1px solid {PALETTE['border']};
  border-radius: 14px;
  padding: .8rem 1rem;
  margin-bottom: 1rem;
}}

[data-testid="stMetric"] {{
  background: rgba(255,255,255,.92);
  border: 1px solid {PALETTE['border']};
  padding: .7rem .85rem;
  border-radius: 14px;
}}

.stButton > button, .stDownloadButton > button {{
  border-radius: 10px;
  min-height: 2.55rem;
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: .3rem;
  background: #EDF2F4;
  border-radius: 12px;
  padding: .3rem;
}}
.stTabs [data-baseweb="tab"] {{
  border-radius: 9px;
}}

[data-testid="stExpander"] {{
  border: 1px solid {PALETTE['border']};
  border-radius: 12px;
  background: rgba(255,255,255,.78);
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, *, eyebrow: str = "Health-economic modelling") -> None:
    st.markdown(
        f"""
<div class="eep-hero">
  <div class="eyebrow">{html.escape(eyebrow)}</div>
  <h1>{html.escape(title)}</h1>
  <p>{html.escape(subtitle)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str, *, kicker: str | None = None) -> None:
    kicker_html = f'<div class="eep-kicker">{html.escape(kicker)}</div>' if kicker else ""
    st.markdown(
        f'<div class="eep-card">{kicker_html}<h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


def coloured_block(title: str, body: str, *, tone: str = "blue", kicker: str | None = None) -> None:
    tone = tone if tone in {"blue", "teal", "green", "amber", "red"} else "blue"
    kicker_html = f'<div class="eep-kicker">{html.escape(kicker)}</div>' if kicker else ""
    st.markdown(
        f'<div class="eep-block eep-block-{tone}">{kicker_html}<h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


def workflow_step(number: int, title: str, body: str) -> None:
    st.markdown(
        f'<div class="eep-step"><strong>{number}. {html.escape(title)}</strong><br><span>{html.escape(body)}</span></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, *, tone: str = "neutral") -> str:
    tone = tone if tone in {"green", "amber", "blue", "teal", "red", "neutral"} else "neutral"
    return f'<span class="eep-badge eep-badge-{tone}">{html.escape(text)}</span>'


def status_bar(items: Iterable[tuple[str, str]]) -> None:
    """Render labelled status pills. Each item is (label, tone)."""
    pills = "".join(badge(label, tone=tone) for label, tone in items)
    st.markdown(f'<div class="eep-statusbar">{pills}</div>', unsafe_allow_html=True)
