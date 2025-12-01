"""
Annotate module
"""

import contextlib
import datetime
import os
import re
import tempfile

from urllib.parse import urlparse
from urllib.request import urlopen, Request

from tqdm.auto import tqdm
from txtai import Embeddings, LLM
from txtai.pipeline import Pipeline, Segmentation
from txtmarker.factory import Factory


class Annotate(Pipeline):
    """
    Automatically annotates papers using LLMs.
    """

    def __init__(self, llm, maxlength=10000, **kwargs):
        """
        Creates a new Annotation pipeline.

        Args:
            llm: LLM path
            maxlength: maximum prompt length
            kwargs: additional LLM args
        """

        # Create LLM pipeline
        self.llm = LLM(llm, **kwargs)
        self.maxlength = maxlength

        # Create segmentation pipeline
        self.segment = Segmentation(sentences=True, cleantext=False)

        # Create highlighter instance
        self.highlighter = Factory.create("pdf")

    def __call__(self, path, output=None, keywords=None, progress=True):
        """
        Reads an input file, generates annotations using LLMs and writes a new file with the
        annotations applied.

        Each annotation also has a generated short topic that renders in the article margins.

        Args:
            path: path to input file
            output: optional path to output file, otherwise this generates a temporary file path
            keywords: optional list of keywords used for the annotation search, otherwise keywords are LLM-generated
            progress: progress bar if True, defaults to True

        Returns:
            output file path
        """

        # Prepare input parameters
        path = self.path(path)
        output = self.output(path, output)

        # Extract text using txtmarker to match formatting exactly
        pages = [text for _, _, text in tqdm(self.highlighter.pages(path), desc="Extracting page text", disable=not progress)]

        # Get title and search keywords
        title = self.title(pages[0], progress)
        keywords = keywords if keywords else self.keywords(pages[0], progress)

        # Get annotations for input file
        annotations = self.annotations(pages, title, keywords, progress)

        # Add topics for each annotation
        annotations = self.topics(annotations, progress)

        # Add the annotation date
        annotations = [(f"Annotated {datetime.datetime.today().strftime('%b-%d-%y %H:%M:%S')}", title)] + annotations

        # Apply the annotations - escape regex since txtmarker supports regular expressions
        self.highlighter.highlight(path, output, [(x, re.escape(y)) for x, y in annotations])

        return output

    def path(self, path):
        """
        Gets the input path. This method downloads path if it's a http(s) url. Otherwise, the input path is returned.

        Args:
            path: input path or url

        Returns:
            local file path
        """

        # Retrieve and write data to temporary file
        if urlparse(path).scheme in ("http", "https"):
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as output:
                with contextlib.closing(urlopen(Request(path))) as connection:
                    output.write(connection.read())

                # Set to output file name
                path = output.name

        return path

    def output(self, path, output):
        """
        Gets the output path. If output is None, a output path is generated using path. Otherise, output is returned.

        Args:
            path: path to input file
            output: path to output file

        Returns:
            output path
        """

        if not output:
            components = os.path.splitext(path)
            output = f"{components[0]}-annotated{components[1]}"

        return output

    def title(self, text, progress):
        """
        Extracts title from text using a LLM prompt.

        Args:
            text: text to analyze
            progress: show progress bar if True

        Returns:
            title
        """

        prompt = f"""
        Extract the paper title from the following text. Only return the title.

        {text}
        """

        result = None
        for x in tqdm([prompt], desc="Extracting title", disable=not progress):
            result = self.llm([{"role": "user", "content": x}], maxlength=2048, stripthink=True)

        return result

    def keywords(self, text, progress):
        """
        Generates keywords from text using a LLM prompt.

        Args:
            text: text to analyze
            progress: show progress bar if True

        Returns:
            keywords
        """

        prompt = f"""
        Generate the best highly descriptive keywords for the paper. Only return the keywords as comma separated.

        {text}
        """

        result = None
        for x in tqdm([prompt], desc="Generating keywords", disable=not progress):
            result = self.llm([{"role": "user", "content": x}], maxlength=2048, stripthink=True)

        return result

    def annotations(self, pages, title, keywords, progress):
        """
        Generates a list of annotations. This method builds an embeddings index for each page and finds the best
        matching sentences for the input keywords.

        Args:
            pages: list of pages
            title: extracted title
            keywords: keywords to search for
            progress: show progress bar if True

        Returns:
            annotations
        """

        annotations = []
        for x, page in enumerate(tqdm(pages, desc="Generating annotations", disable=not progress)):
            # Build embeddings index for page
            embeddings = self.embeddings(page)

            queries = keywords if isinstance(keywords, list) else [x.strip() for x in keywords.split(",")]
            results = {}
            for result in embeddings.batchsearch(queries, 10):
                # Add each result if it meets the following conditions
                #  - Not a current result or a higher scoring result for same text AND
                #  - Not the title AND
                #  - Score is greater than the min threshold

                for x in result:
                    text, score, prevscore = (x["text"], x["score"], results.get(x["text"]))
                    if (text not in results or score > prevscore) and title not in text.replace("\n", " ") and score >= 0.1:
                        results[text] = x["score"]

            # Add top 5 best scoring results as annotations
            for text, _ in sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]:
                annotations.append(text)

        return annotations

    def topics(self, annotations, progress):
        """
        Generates topics for each annotation.

        Args:
            annotations: list of annotations
            progress: show progress bar if True

        Returns:
            [(topic, annotation)]
        """

        prompt = """
        Create a simple, concise topic name in less than 5 words for the following text. Only return the topic name.

        Text:
        {text}
        """

        # Build prompts
        prompts = [prompt.format(text=x) for x in annotations]

        # Generate topics
        topics = []
        for prompt in tqdm(prompts, desc="Generating topics", disable=not progress):
            # Generate topic
            topic = self.llm([{"role": "user", "content": prompt}], maxlength=self.maxlength, stripthink=True)

            # Clean topic and append
            topics.append(re.sub(r"[^\x00-\x7f]", r"", topic))

        # Return list of (topic, annotation) pairs
        return list(zip(topics, annotations))

    def embeddings(self, page):
        """
        Splits page into sentences and builds an embeddings index.

        Args:
            page: page of text

        Returns:
            embeddings index
        """

        # Split into sentences, require a minimum number of words
        sentences = [x for x in self.segment(page) if x.strip() and len(x.split()) >= 5]

        # Build embeddings index for sentences
        embeddings = Embeddings(content=True, hybrid=True)
        embeddings.index(sentences)

        return embeddings
