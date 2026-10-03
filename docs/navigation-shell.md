# Navigation shell

The Streamlit entrypoint (`app.py`) acts as a router using `st.Page` and `st.navigation`. The landing page is explicitly registered as **Home**, so the navigation label is independent of the entrypoint filename used by Streamlit Community Cloud.

The router groups the application into Start, Economic models, and Review & reproducibility sections. The shared visual design system is applied around every registered page so modeller pages inherit the same palette and structural styling without changing analytical engines.
