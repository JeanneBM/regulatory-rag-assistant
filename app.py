import streamlit as st
from rag import answer, load_corpus, retrieve

st.set_page_config(page_title="Regulatory RAG Assistant")
st.title("Regulatory RAG Assistant")
st.caption("AI Act and DORA · original published acts · research PoC")
st.info("Sources are pinned snapshots, not a guarantee of current law. "
        "Amendments, annexes and delegated acts are outside this first release.")
act = st.selectbox("Source filter", ["All", "AI_ACT", "DORA"])
question = st.text_area("Question", placeholder="What does DORA say about ICT risk management?")
if st.button("Search", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving evidence"):
            evidence = retrieve(question, load_corpus(), None if act == "All" else act)
            result = answer(question, evidence)
        st.write("Status:", result["status"])
        if result["status"] == "retrieval_only":
            st.caption("Configure Azure OpenAI to enable generated answers. English queries work best with this index.")
        for claim in result["claims"]:
            st.write(claim["statement"])
            st.text(claim["quote"])
            st.caption("Source: " + claim["source_id"])
        for chunk in evidence:
            with st.expander(f'{chunk["id"]} · score {chunk["score"]:.3f}'):
                st.text(chunk["text"])
                st.link_button("Official source", chunk["url"])
                st.caption(f'Fetched: {chunk["fetched_at"]} · SHA256: {chunk["sha256"]}')
    except Exception:
        st.error("Request failed. Check corpus import and provider configuration in your local terminal.")
