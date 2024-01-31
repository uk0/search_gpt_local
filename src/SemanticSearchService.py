import os

import openai
import pandas as pd
import re

from openai.embeddings_utils import cosine_similarity
from website.sender import Sender, MSG_TYPE_SEARCH_STEP

from Util import setup_logger
from NLPUtil import num_tokens_from_string

logger = setup_logger('SemanticSearchService')
class BatchOpenAISemanticSearchService:
    def __init__(self, config, sender: Sender = None):
        self.config = config
        openai.api_key = config.get('llm_service').get('openai_api').get('api_key')
        self.sender = sender

    @staticmethod
    def batch_call_embeddings(texts, chunk_size=1000):
        texts = [text.replace("\n", " ") for text in texts]
        embeddings = []
        response = None
        for i in range(0, len(texts), chunk_size):
            if os.environ.get('LOCAL_LLM') is not None:
                response = openai.Embedding.create(
                    input=texts[i: i + chunk_size],
                    engine="text-embedding-ada-002",
                )
            else:
                response = openai.Embedding.create(
                    input=texts[i: i + chunk_size],
                    engine="moe",
                    # default embedding of faiss-openai,
                    api_base="http://localhost:8080/v1"
                )
            embeddings += [r["embedding"] for r in response["data"]]
        return embeddings

    @staticmethod
    def compute_embeddings_for_text_df(text_df: pd.DataFrame):
        """Compute embeddings for a text_df and return the text_df with the embeddings column added."""
        print(f'compute_embeddings_for_text_df() len(texts): {len(text_df)}')
        text_df['text'] = text_df['text'].apply(lambda x: x.replace("\n", " "))
        text_df['embedding'] = BatchOpenAISemanticSearchService.batch_call_embeddings(text_df['text'].tolist())
        return text_df

    def search_related_source(self, text_df: pd.DataFrame, target_text, n=30):
        if not self.config.get('source_service').get('is_use_source'):
            col = ['name', 'url', 'url_id', 'snippet', 'text', 'similarities', 'rank', 'docno']
            return pd.DataFrame(columns=col)

        if self.sender is not None:
            self.sender.send_message(msg_type=MSG_TYPE_SEARCH_STEP, msg="Searching from extracted text")
        print(f'search_similar() text: {target_text}')
        embedding = BatchOpenAISemanticSearchService.batch_call_embeddings([target_text])[0]
        text_df = BatchOpenAISemanticSearchService.compute_embeddings_for_text_df(text_df)
        text_df['similarities'] = text_df['embedding'].apply(lambda x: cosine_similarity(x, embedding))
        result_df = text_df.sort_values('similarities', ascending=False).head(n)
        result_df['rank'] = range(1, len(result_df) + 1)
        result_df['docno'] = range(1, len(result_df) + 1)
        return result_df

    @staticmethod
    def post_process_gpt_input_text_df(gpt_input_text_df, prompt_token_limit):
        # clean out of prompt texts for existing [1], [2], [3]... in the source_text for response output stability
        gpt_input_text_df['text'] = gpt_input_text_df['text'].apply(lambda x: re.sub(r'\[[0-9]+\]', '', x))
        # length of char and token
        gpt_input_text_df['len_text'] = gpt_input_text_df['text'].apply(lambda x: len(x))
        gpt_input_text_df['len_token'] = gpt_input_text_df['text'].apply(lambda x: num_tokens_from_string(x))

        gpt_input_text_df['cumsum_len_text'] = gpt_input_text_df['len_text'].cumsum()
        gpt_input_text_df['cumsum_len_token'] = gpt_input_text_df['len_token'].cumsum()

        max_rank = gpt_input_text_df[gpt_input_text_df['cumsum_len_token'] <= prompt_token_limit]['rank'].max() + 1
        gpt_input_text_df['in_scope'] = gpt_input_text_df[
                                            'rank'] <= max_rank  # In order to get also the row slightly larger than prompt_length_limit
        # reorder url_id with url that in scope.
        url_id_list = gpt_input_text_df['url_id'].unique()
        url_id_map = dict(zip(url_id_list, range(1, len(url_id_list) + 1)))
        gpt_input_text_df['url_id'] = gpt_input_text_df['url_id'].map(url_id_map)
        return gpt_input_text_df
