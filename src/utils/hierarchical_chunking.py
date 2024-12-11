


def count_tokens(text: list[str] | None, tokenizer):
    if text is None:
        return 0
    elif isinstance(text, list):
        total = sum(count_tokens(t, tokenizer) for t in text)
        return total
    return len(tokenizer.tokenize(text, max_length=None))

def make_splitter(tokenizer, chunk_size):
    return semchunk.chunkerify(tokenizer, chunk_size)

def doc_chunk_length(doc_chunk: DocChunk, tokenizer):
    text_length = count_tokens(doc_chunk.text, tokenizer)
    headings_length = count_tokens(doc_chunk.meta.headings, tokenizer)
    captions_length = count_tokens(doc_chunk.meta.captions, tokenizer)
    total = text_length + headings_length + captions_length
    return {"total": total, "text": text_length, "other": total - text_length}

def make_chunk_from_doc_items(
    doc_chunk: DocChunk, window_text: str, window_start: int, window_end: int
) -> DocChunk:
    meta = DocMeta(
        doc_items=doc_chunk.meta.doc_items[window_start:window_end + 1],
        headings=doc_chunk.meta.headings,
        captions=doc_chunk.meta.captions,
    )
    new_chunk = DocChunk(text=window_text, meta=meta)
    return new_chunk

def merge_text(t1: str, t2: str) -> str:
    if t1 == "":
        return t2
    elif t2 == "":
        return t1
    else:
        return t1 + "\n" + t2

def split_by_doc_items(doc_chunk: DocChunk, tokenizer, chunk_size: int) -> List[DocChunk]:
    if doc_chunk.meta.doc_items is None or len(doc_chunk.meta.doc_items) <= 1:
        return [doc_chunk]
    length = doc_chunk_length(doc_chunk, tokenizer)
    if length["total"] <= chunk_size:
        return [doc_chunk]
    else:
        chunks = []
        window_start = 0
        window_end = 0
        window_text = ""
        window_text_length = 0
        other_length = length["other"]
        l = len(doc_chunk.meta.doc_items)
        while window_end < l:
            doc_item = doc_chunk.meta.doc_items[window_end]
            text = doc_item.text
            text_length = count_tokens(text, tokenizer)
            if (
                text_length + window_text_length + other_length < chunk_size
                and window_end < l - 1
            ):
                window_end += 1
                window_text_length += text_length
                window_text = merge_text(window_text, text)
            elif text_length + window_text_length + other_length < chunk_size:
                window_text = merge_text(window_text, text)
                new_chunk = make_chunk_from_doc_items(
                    doc_chunk, window_text, window_start, window_end
                )
                chunks.append(new_chunk)
                window_end = l
            elif window_start == window_end:
                window_text = merge_text(window_text, text)
                new_chunk = make_chunk_from_doc_items(
                    doc_chunk, window_text, window_start, window_end
                )
                chunks.append(new_chunk)
                window_start = window_end + 1
                window_end = window_start
                window_text = ""
                window_text_length = 0
            else:
                new_chunk = make_chunk_from_doc_items(
                    doc_chunk, window_text, window_start, window_end - 1
                )
                chunks.append(new_chunk)
                window_start = window_end
                window_text = ""
                window_text_length = 0

        return chunks

def split_using_plain_text(
    doc_chunk: DocChunk,
    tokenizer,
    plain_text_splitter,
    chunk_size: int,
) -> List[DocChunk]:
    lengths = doc_chunk_length(doc_chunk, tokenizer)
    if lengths["total"] <= chunk_size:
        return [doc_chunk]
    else:
        available_length = chunk_size - lengths["other"]
        if available_length <= 0:
            raise ValueError(
                "Headers and captions for this chunk are longer than the total amount of size for the chunk. This is not supported now."
            )
        text = doc_chunk.text
        segments = plain_text_splitter.chunk(text)
        chunks = []
        for s in segments:
            new_chunk = DocChunk(text=s, meta=doc_chunk.meta)
            chunks.append(new_chunk)
        return chunks

def merge_chunks_with_matching_metadata(chunks, tokenizer, chunk_size):
    output_chunks = []
    window_start = 0
    window_end = 0
    l = len(chunks)
    while window_end < l:
        chunk = chunks[window_end]
        lengths = doc_chunk_length(chunk, tokenizer)
        headings_and_captions = (chunk.meta.headings, chunk.meta.captions)
        if window_start == window_end:
            current_headings_and_captions = headings_and_captions
            window_text = chunk.text
            window_other_length = lengths["other"]
            window_text_length = lengths["text"]
            window_items = chunk.meta.doc_items
            window_end += 1
            first_chunk_of_window = chunk
        elif (
            headings_and_captions == current_headings_and_captions
            and window_text_length + window_other_length + lengths["text"] <= chunk_size
        ):
            window_text = merge_text(window_text, chunk.text)
            window_text_length += lengths["text"]
            window_items = window_items + chunk.meta.doc_items
            window_end += 1
        else:
            if window_start + 1 == window_end:
                output_chunks.append(first_chunk_of_window)
            else:
                new_meta = DocMeta(
                    doc_items=window_items,
                    headings=headings_and_captions[0],
                    captions=headings_and_captions[1],
                )
                new_chunk = DocChunk(text=window_text, meta=new_meta)
                output_chunks.append(new_chunk)
            window_start = window_end

    return output_chunks

def merge_chunks_with_mismatching_metadata(chunks, *_):
    return chunks

def merge_chunks(chunks, tokenizer, chunk_size):
    initial_merged_chunks = merge_chunks_with_matching_metadata(
        chunks, tokenizer, chunk_size
    )
    final_merged_chunks = merge_chunks_with_mismatching_metadata(
        initial_merged_chunks, tokenizer, chunk_size
    )
    return final_merged_chunks

def adjust_chunks_for_fixed_size(doc, original_chunks, tokenizer, splitter, chunk_size):
    chunks_after_splitting_by_items = []
    for chunk in original_chunks:
        chunk_split_by_doc_items = split_by_doc_items(chunk, tokenizer, chunk_size)
        chunks_after_splitting_by_items.extend(chunk_split_by_doc_items)
    chunks_after_splitting_recursively = []
    for chunk in chunks_after_splitting_by_items:
        chunk_split_recursively = split_using_plain_text(
            chunk, tokenizer, splitter, chunk_size
        )
        chunks_after_splitting_recursively.extend(chunk_split_recursively)
    chunks_after_merging = merge_chunks(
        chunks_after_splitting_recursively, tokenizer, chunk_size
    )
    return chunks_after_merging

class MaxTokenLimitingChunkerWithMerging(BaseChunker):
    inner_chunker: BaseChunker = HierarchicalChunker()
    max_tokens: PositiveInt = 512
    embedding_model_id: str

    def chunk(self, dl_doc: DoclingDocument, **kwargs) -> Iterator[BaseChunk]:
        preliminary_chunks = self.inner_chunker.chunk(dl_doc=dl_doc, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_id)
        splitter = make_splitter(tokenizer, self.max_tokens)
        output_chunks = adjust_chunks_for_fixed_size(
            dl_doc, preliminary_chunks, tokenizer, splitter, self.max_tokens
        )
        return iter(output_chunks)