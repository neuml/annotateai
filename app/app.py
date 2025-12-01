"""
This application annotates PDFs at a given URL or file path.

Requires the following packages
  pip install annotateai streamlit streamlit-pdf-viewer
"""

import os

from urllib.parse import urlparse

import streamlit as st

from streamlit_pdf_viewer import pdf_viewer

from txtai import Embeddings

from annotateai import Annotate


class Application:
    """
    Main application.
    """

    def __init__(self):
        """
        Creates a new application.
        """

        # Search for additional allowed environment variables
        kwargs = {x: int(os.environ[x]) for x in ["n_ctx"] if x in os.environ}

        # Annotation instance
        self.annotate = Annotate(
            llm=os.environ.get("LLM", "Qwen/Qwen3-4B-Instruct-2507"), maxlength=int(os.environ.get("MAXLENGTH", 10000)), **kwargs
        )

        # Embeddings database for search (lazy loaded)
        self.embeddings = None

    def run(self):
        """
        Main rendering logic.
        """

        # Example papers
        examples = {
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": "https://arxiv.org/pdf/2005.11401",
            "HunyuanVideo: A Systematic Framework For Large Video Generative Models": "https://arxiv.org/pdf/2412.03603v2",
            "OpenDebateEvidence: A Massive-Scale Argument Mining and Summarization Dataset": "https://arxiv.org/pdf/2406.14657",
        }

        # UI components
        url, selected, download, render = st.empty(), st.empty(), st.empty(), st.empty()

        # Get examples
        selected = selected.pills("**or try these examples**", examples, key="example", on_change=self.onchange)
        selected = st.session_state.get("selected")

        # Create URL input using selected example, if applicable
        url = url.text_input("**URL / Local File Path / Search**", value=examples.get(selected, ""))

        # Annotate the URL
        if url:
            # Check if URL is valid, otherwise run an embeddings search
            url = self.validate(url)

            # Build the annotation file for URL
            with st.spinner(f"Generating annotations for {url}"):
                # Get the annotated output
                output = self.build(url)

            # Get url file name
            _, name = os.path.split(url)
            name = name if name.lower().endswith(".pdf") else f"{name}.pdf"

            # Download file
            with open(output, "rb") as outfile:
                download.download_button(label=f"Download {name}", data=outfile, file_name=name, mime="application/pdf", type="primary")

            # File previews
            with render.container():
                # Limit file previews to files <= 20 MB due to data url size limitations
                if os.path.getsize(output) >= 20 * 1024 * 1024:
                    st.info("File too big to preview. Click download to view.")
                else:
                    st.divider()
                    pdf_viewer(output, render_text=True)

    def onchange(self):
        """
        Sets the selected example and clears the UI component.
        """

        st.session_state.selected = st.session_state.example
        st.session_state.example = None
        st.session_state.url = None

    def validate(self, url):
        """
        Checks if input is a url or local file path. Otherwise, this runs a search and returns
        the url for the top result.

        Args:
            url: input url, local file path or search query

        Returns:
            url
        """

        # Check if this is a URL or local file path
        if urlparse(url).scheme in ("http", "https") or os.path.exists(url):
            return url

        # Lazy load of txtai-arxiv embeddings database
        if not self.embeddings:
            with st.spinner("Loading txtai-arxiv embeddings index for search"):
                self.embeddings = Embeddings().load(provider="huggingface-hub", container="neuml/txtai-arxiv")

        # Get top matching article
        result = self.embeddings.search(url, 1)[0]
        title = result["text"].split("\n")[0].replace("\n", " ")

        st.toast(f"Ran search for {url} and using top match `{title}`")
        return f"https://arxiv.org/pdf/{result['id']}"

    # pylint: disable=E0213
    @st.cache_data(show_spinner=False)
    def build(_self, url):
        """
        Builds an annotation for url.

        Args:
            url: url to annotate

        Returns:
            annotated file path
        """

        return _self.annotate(url)


@st.cache_resource(show_spinner="Initializing application...")
def create():
    """
    Creates and caches a Streamlit application.

    Returns:
        Application
    """

    return Application()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    st.set_page_config(
        page_title="Annotate papers using LLMs",
        page_icon="📝",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.markdown("## 📝 Annotate papers using LLMs")

    st.markdown(
        """
This application automatically annotates papers using LLMs.

`Annotate URLs or local file paths, if found. Otherwise, the top result from the txtai-arxiv embeddings database is returned for the input.`

Try PDFs from [arXiv](https://arxiv.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/),
[bioRxiv](https://www.biorxiv.org/) or [medRxiv](https://www.medrxiv.org/)!
"""
    )

    # Create and run application
    app = create()
    app.run()
