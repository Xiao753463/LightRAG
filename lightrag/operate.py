from __future__ import annotations

import asyncio
from datetime import datetime
import json
import re
import os
from typing import Any, AsyncIterator, Union
from collections import Counter, defaultdict

from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    statistic_data,
    get_conversation_turns,
    verbose_debug,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
import time
from dotenv import load_dotenv
chunk_entity_relation_graph: BaseGraphStorage
# Load environment variables
load_dotenv(override=True)


def chunking_by_token_size(
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(
                    chunk, model_name=tiktoken_model)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(
                    chunk, model_name=tiktoken_model)
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = decode_tokens_by_tiktoken(
                            _tokens[start: start + max_token_size],
                            model_name=tiktoken_model,
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = decode_tokens_by_tiktoken(
                tokens[start: start + max_token_size], model_name=tiktoken_model
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(
        description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None

    # Clean and validate entity name
    entity_name = clean_str(record_attributes[1]).strip('"')
    if not entity_name.strip():
        logger.warning(
            f"Entity extraction error: empty entity name in: {record_attributes}"
        )
        return None

    # Clean and validate entity type
    entity_type = clean_str(record_attributes[2]).strip('"')
    if not entity_type.strip() or entity_type.startswith('("'):
        logger.warning(
            f"Entity extraction error: invalid entity type in: {record_attributes}"
        )
        return None

    # Clean and validate description
    entity_description = clean_str(record_attributes[3]).strip('"')
    if not entity_description.strip():
        logger.warning(
            f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
        )
        return None

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=chunk_key,
        metadata={"created_at": time.time()},
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1]).strip('"')
    target = clean_str(record_attributes[2]).strip('"')
    edge_description = clean_str(record_attributes[3]).strip('"')
    edge_keywords = clean_str(record_attributes[4]).strip('"')
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1].strip('"'))
        if is_float_regex(record_attributes[-1])
        else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        metadata={"created_at": time.time()},
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(
                already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"]
               for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 0.0 if missing
            already_weights.append(already_edge.get("weight", 0.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    split_string_by_multi_markers(
                        already_edge["source_id"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.append(already_edge["description"])

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    # Process edges_data with None checks
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(
            set(
                [dp["description"]
                    for dp in edges_data if dp.get("description")]
                + already_description
            )
        )
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(
            set(
                [dp["keywords"] for dp in edges_data if dp.get("keywords")]
                + already_keywords
            )
        )
    )
    source_id = GRAPH_FIELD_SEP.join(
        set(
            [dp["source_id"] for dp in edges_data if dp.get("source_id")]
            + already_source_ids
        )
    )

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "entity_id": need_insert_id,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": "UNKNOWN",
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
) -> None:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    enable_llm_cache_for_entity_extract: bool = global_config[
        "enable_llm_cache_for_entity_extract"
    ]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entity_continue_extraction"].format(
        **context_base)
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _user_llm_func_with_cache(
        input_text: str, history_messages: list[dict[str, str]] = None
    ) -> str:
        if enable_llm_cache_for_entity_extract and llm_response_cache:
            if history_messages:
                history = json.dumps(history_messages, ensure_ascii=False)
                _prompt = history + "\n" + input_text
            else:
                _prompt = input_text

            # TODO： add cache_type="extract"
            arg_hash = compute_args_hash(_prompt)
            cached_return, _1, _2, _3 = await handle_cache(
                llm_response_cache,
                arg_hash,
                _prompt,
                "default",
                cache_type="extract",
            )
            if cached_return:
                logger.debug(f"Found cache for {arg_hash}")
                statistic_data["llm_cache"] += 1
                return cached_return
            statistic_data["llm_call"] += 1
            if history_messages:
                res: str = await use_llm_func(
                    input_text, history_messages=history_messages
                )
            else:
                res: str = await use_llm_func(input_text)
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type="extract",
                ),
            )
            return res

        if history_messages:
            return await use_llm_func(input_text, history_messages=history_messages)
        else:
            return await use_llm_func(input_text)

    async def _process_extraction_result(result: str, chunk_key: str):
        """Process a single extraction result (either initial or gleaning)
        Args:
            result (str): The extraction result to process
            chunk_key (str): The chunk key for source tracking
        Returns:
            tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
        """
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        records = split_string_by_multi_markers(
            result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        return maybe_nodes, maybe_edges

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        # Get initial extraction
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await _user_llm_func_with_cache(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

        # Process initial extraction
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result, chunk_key
        )

        # Process additional gleaning results
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await _user_llm_func_with_cache(
                continue_prompt, history_messages=history
            )

            history += pack_user_ass_to_openai_messages(
                continue_prompt, glean_result)

            # Process gleaning result separately
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result, chunk_key
            )

            # Merge results
            for entity_name, entities in glean_nodes.items():
                maybe_nodes[entity_name].extend(entities)
            for edge_key, edges in glean_edges.items():
                maybe_edges[edge_key].extend(edges)

            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await _user_llm_func_with_cache(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"  Chunk {processed_chunks}/{total_chunks}: extracted {entities_count} entities and {relations_count} relationships (deduplicated)"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)
        return dict(maybe_nodes), dict(maybe_edges)

    tasks = [_process_single_content(c) for c in ordered_chunks]
    results = await asyncio.gather(*tasks)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    from .kg.shared_storage import get_graph_db_lock

    graph_db_lock = get_graph_db_lock(enable_logging=False)

    # Ensure that nodes and edges are merged and upserted atomically
    async with graph_db_lock:
        all_entities_data = await asyncio.gather(
            *[
                _merge_nodes_then_upsert(
                    k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        )

        all_relationships_data = await asyncio.gather(
            *[
                _merge_edges_then_upsert(
                    k[0], k[1], v, knowledge_graph_inst, global_config
                )
                for k, v in maybe_edges.items()
            ]
        )

    if not (all_entities_data or all_relationships_data):
        log_message = "Didn't extract any entities and relationships."
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)
        return

    if not all_entities_data:
        log_message = "Didn't extract any entities"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)
    if not all_relationships_data:
        log_message = "Didn't extract any relationships"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    log_message = f"Extracted {len(all_entities_data)} entities and {len(all_relationships_data)} relationships (deduplicated)"
    logger.info(log_message)
    if pipeline_status is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)
    verbose_debug(
        f"New entities:{all_entities_data}, relationships:{all_relationships_data}"
    )
    verbose_debug(f"New relationships:{all_relationships_data}")

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "entity_name": dp["entity_name"],
                "entity_type": dp["entity_type"],
                "content": f"{dp['entity_name']}\n{dp['description']}",
                "source_id": dp["source_id"],
                "metadata": {
                    "created_at": dp.get("metadata", {}).get("created_at", time.time())
                },
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "keywords": dp["keywords"],
                "content": f"{dp['src_id']}\t{dp['tgt_id']}\n{dp['keywords']}\n{dp['description']}",
                "source_id": dp["source_id"],
                "metadata": {
                    "created_at": dp.get("metadata", {}).get("created_at", time.time())
                },
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            "low_level_keywords is empty, switching from %s mode to global mode",
            query_param.mode,
        )
        query_param.mode = "global"
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            "high_level_keywords is empty, switching from %s mode to local mode",
            query_param.mode,
        )
        query_param.mode = "local"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build context
    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )
    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            "low_level_keywords is empty, switching from %s mode to global mode",
            query_param.mode,
        )
        query_param.mode = "global"
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            "high_level_keywords is empty, switching from %s mode to local mode",
            query_param.mode,
        )
        query_param.mode = "local"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build context
    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )
    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(param.mode, text, cache_type="keywords")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return keywords_data["high_level_keywords"], keywords_data[
                "low_level_keywords"
            ]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 3. Process conversation history
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text, examples=examples, language=language, history=history_context
    )

    len_of_prompts = len(encode_string_by_tiktoken(kw_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    # 5. Call the LLM for keyword extraction
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, keyword_extraction=True)
    # 6. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        keywords_data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 7. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="keywords",
            ),
        )
    print("hl_keywords: ", hl_keywords, " ll_keywords: ", ll_keywords)
    return hl_keywords, ll_keywords


async def kg_query_plus(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None
) -> str | AsyncIterator[str]:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Extract keywords using extract_keywords_only function which already supports conversation history
    keywords = await extract_keywords_only_plus(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"keywords: {keywords}")

    # Handle empty keywords
    if keywords == []:
        logger.warning("keywords is empty")
        return PROMPTS["fail_response"]

    keywords_str = ", ".join(keywords) if keywords else ""
    global chunk_entity_relation_graph
    chunk_entity_relation_graph = knowledge_graph_inst
    # Build context
    context = await _build_query_context(
        keywords_str,
        "",
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )
    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        tools=[],
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response


async def get_function_call_results(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    knowledge_graph_inst: BaseGraphStorage,
    sys_prompt: str | None = None,
) -> tuple[list[str], list[str]]:

    use_model_func = global_config["llm_model_func"]
    # 定義 function calling tool
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_time_ranges",
                "description": "根據使用者提出的市場問題，自動萃取出時間區間（開始與結束日期）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_ranges": {
                            "type": "array",
                            "description": "時間區間列表",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_date": {
                                        "type": "string",
                                        "description": "區間的開始日期，格式為 YYYY-MM-DD"
                                    },
                                    "end_date": {
                                        "type": "string",
                                        "description": "區間的結束日期，格式為 YYYY-MM-DD"
                                    }
                                },
                                "required": ["start_date", "end_date"]
                            }
                        }
                    },
                    "required": ["time_ranges"]
                }
            }
        }

        # {
        #     "type": "function",
        #     "function": {
        #         "name": "get_entity_statistics",
        #         "description": "當使用者想查詢某一類節點（品牌、類型、商品、服務）中哪些項目『出現頻率較高』或『情感評價較好／較差』時，請使用此函數。使用時，務必根據使用者問題中的分析重點，從下列四個節點類型中擇一填入 node_type：品牌、類型、商品、服務。⚠️ 請勿使用其他字詞，僅能從上述五個值中選擇，且需完全對應。",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "node_type": {
        #                     "type": "string",
        #                     "enum": ["品牌", "類型", "商品", "服務"],
        #                     "description": "欲查詢的節點類型，只能從以下四個字詞中擇一填入，且必須完全一致（不得翻譯、拼錯或創造）：品牌、類型、商品、服務"
        #                 },
        #                 "start_time": {
        #                     "type": "string",
        #                     "format": "date-time",
        #                     "description": "查詢的起始日期，使用ISO 8601格式，如2025-01-01"
        #                 },
        #                 "end_time": {
        #                     "type": "string",
        #                     "format": "date-time",
        #                     "description": "查詢的結束日期，使用ISO 8601格式，如2025-12-31"
        #                 }
        #             },
        #             "required": ["node_type"]
        #         }
        #     }
        # },
        # # {
        # #     "type": "function",
        # #     "function": {
        # #         "name": "get_facet_statistics",
        # #         "description": "當使用者想查詢某一類型(商品、服務)的構面中哪些項目『出現頻率較高』、『情感評價較好／較差』或『標竿商品為何』時，請使用此函數。使用時，務必根據使用者問題中的分析重點，從下列兩個構面類型中擇一填入或是不填 facet_type：商品、服務。⚠️ 請勿使用其他字詞，僅能從上述兩個值中選擇，且需完全對應。",
        # #         "parameters": {
        # #             "type": "object",
        # #             "properties": {
        # #                 "facet_type": {
        # #                     "type": "string",
        # #                     "enum": ["商品", "服務"],
        # #                     "description": "欲查詢的構面所屬的類型，只能從以下兩個字詞中擇一填入，且必須完全一致（不得翻譯、拼錯或創造）：商品、服務"
        # #                 },
        # #                 "start_time": {
        # #                     "type": "string",
        # #                     "format": "date-time",
        # #                     "description": "查詢的起始日期，使用ISO 8601格式，如2025-01-01"
        # #                 },
        # #                 "end_time": {
        # #                     "type": "string",
        # #                     "format": "date-time",
        # #                     "description": "查詢的結束日期，使用ISO 8601格式，如2025-12-31"
        # #                 }
        # #             }
        # #         }
        # #     }
        # # },
        # # {
        # #     "type": "function",
        # #     "function": {
        # #         "name": "get_facet_influence",
        # #         "description": "當使用者想查詢哪項商品在某個構面中的表現較好時，請使用此函數。使用時，務必根據使用者問題中的分析重點，了解目標構面",
        # #         "parameters": {
        # #             "type": "object",
        # #             "properties": {
        # #                 "facet_name": {
        # #                     "type": "string",
        # #                     "description": "欲查詢的構面名稱"
        # #                 }
        # #             },
        # #             "required": ["facet_name"]
        # #         }
        # #     }
        # # }

    ]

    # async def get_entity_statistics_for_function_call(node_type: str, start_time: str = Any | None, end_time: str = Any | None):
    #     """
    #     透過給定的 node_type 來提取該節點類型的所有實體的提及次數與情感分數。

    #     Args:
    #     node_type (str): 欲查詢的節點類型，只能從以下五個字詞中擇一填入，且必須完全一致（不得翻譯、拼錯或創造）："品牌"、"類型"、"商品"、"服務"、"構面"

    #     Returns:
    #     list[dict]: 一個含有實體名稱、實體提及次數、實體情感分數的列表
    #     """
    #     entity_statistics = await knowledge_graph_inst.get_entity_statistics(node_type, start_time, end_time)
    #     print(f"entity_statistics: {entity_statistics}")
    #     result = ''
    #     for index, item in enumerate(entity_statistics):
    #         result += f"""[{index+1}] {item[f'{node_type}名稱']} 被提及了 {item['提及次數']
    #                                                                  } 次, 情感比率為 {float(item['情感分數']/item['提及次數'])}\n"""
    #     return result

    # async def get_facet_statistics_for_function_call(facet_type: str = Any | None, start_time: datetime = Any | None, end_time: datetime = Any | None):
    #     """
    #     透過給定的 facet_type 來提取指定類型中所有構面的提及次數與情感分數。

    #     Args:
    #     facet_type (str): 欲查詢的構面類型，只能從以下兩個字詞中擇一填入，且必須完全一致（不得翻譯、拼錯或創造）："商品"、"服務"

    #     Returns:
    #     list[dict]: 一個含有構面類型、構面名稱、構面及次數、實體情感分數的列表
    #     """
    #     start_date = datetime.strptime(start_time, '%Y-%m-%d').date()
    #     end_date = datetime.strptime(end_time, '%Y-%m-%d').date()
    #     facet_statistics = await knowledge_graph_inst.get_facet_statistics(facet_type, start_date, end_date)
    #     result = ''
    #     for index, item in enumerate(facet_statistics):
    #         result += f"[{index+1}] [{item[f'類型']}] {item[f'構面名稱']} 被提及了 {item['提及次數']
    #                                                                       } 次 關注度為 {item[f'關注度']}, 情感比率為 {float(item['情感分數']/item['提及次數'])}\n"
    #     return result

    # async def get_facet_influence_for_function_call(facet_name: str):
    #     """
    #     透過給定的 facet_name 來提取產品在指定構面的影響力。

    #     Args:
    #     facet_name (str): 欲查詢的構面名稱

    #     Returns:
    #     list[dict]: 一個含有構面類型、構面名稱、構面及次數、實體情感分數的列表
    #     """
    #     facet_statistics = await knowledge_graph_inst.get_facet_statistics()
    #     all_facet_names = []
    #     for item in facet_statistics:
    #         all_facet_names.append(item[f'構面名稱'])
    #     query = f"""
    #     使用者輸入的構面名稱是：「{facet_name}」

    #     目前系統中所有構面名稱如下：
    #     {all_facet_names}

    #     請從上述構面中選出最相近的一項（可依語意或拼音相似度判斷）。
    #     若完全找不到相關構面，請輸出「分析結果: 無相似構面」。

    #     請務必**只輸出一行，格式如下**：
    #     分析結果: 最相近構面名稱

    #     ---範例---
    #     分析結果: 口感
    #     """
    #     response = await use_model_func(
    #         query,
    #         tools=[],
    #         functions=[],
    #         system_prompt=sys_prompt,
    #         stream=param.stream,
    #     )
    #     facet = ''
    #     if "分析結果:" in response:
    #         facet = response.split("分析結果:")[-1].strip()
    #     print(f"目標構面: {facet}")
    #     if facet in ['', '無相似構面']:
    #         return "沒有相關資料"
    #     entity_statistics = await knowledge_graph_inst.get_facet_influence(facet)
    #     result = f'{facet} 構面中，影響力排序如下：\n'
    #     for index, item in enumerate(entity_statistics):
    #         result += f"[{index+1}] {item[f'商品']}： **影響力為 {item[f'影響力']}** ，被提及了 {
    #             item['提及次數']} 次, 情感比率為 {float(item['情感分數']/item['提及次數'])} \n"
    #     if result in [f'{facet} 構面中，影響力排序如下：\n', None]:
    #         return "沒有相關資料"
    #     return result

    async def extract_time_ranges(time_ranges):
        return time_ranges
    available_functions = {
        "extract_time_ranges": extract_time_ranges
    }
    prompt = """
    你是一位專業的市場分析助手，請根據使用者提出的市場問題句子，萃取出其中所有涉及的時間區間，並輸出為標準格式以供 function 調用。

    請務必輸出 JSON 格式，格式如下：
    [
        {{
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD"
        }},
        ...
    ]

    請特別注意：
    - 欄位名稱必須為 start_date 和 end_date（請勿使用 start、from、date 等變形）
    - 若句子中沒有明確指定時間，請預設為「近一個月」（今天往前推30天）
    - 若有比較語意，請輸出多組時間區間（例如：上個月 vs 這個月）
    - 日期格式請統一為 YYYY-MM-DD

    今天日期是：{today}
    使用者問題：{query}

    """

    funcall_prompt = prompt.format(
        query=text,
        today=datetime.today().date()
    )

    response = await use_model_func(
        funcall_prompt,
        tools=available_tools,
        functions=available_functions,
        system_prompt=sys_prompt,
        stream=param.stream,
    )
    return response


async def kg_query_plus_with_function_call(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None
) -> str | AsyncIterator[str]:
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    keywords = await extract_keywords_only_plus(
        query, query_param, global_config, hashing_kv
    )
    function_call_results = await get_function_call_results(query, query_param, global_config, knowledge_graph_inst)
    time_range_list = json.loads(function_call_results)
    if keywords == []:
        return PROMPTS["fail_response"]
    keywords_str = ", ".join(keywords)
    # Build context
    context = await _build_query_context(
        keywords_str,
        "",
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        time_range_list
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )
    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")
    response = await use_model_func(
        query,
        tools=[],
        functions=[],
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response


async def extract_keywords_only_plus(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(param.mode, text, cache_type="keywords")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return keywords_data["keywords"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_plus_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_plus_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_plus_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 3. Process conversation history
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction_plus"].format(
        query=text, examples=examples, language=language, history=history_context
    )

    len_of_prompts = len(encode_string_by_tiktoken(kw_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    # 5. Call the LLM for keyword extraction
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, tools=[], functions=[], keyword_extraction=True)
    # 6. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        keywords_data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    keywords = keywords_data.get("keywords", [])
    query_target_types = keywords_data.get("query_target_types", [])

    # 7. Cache only the processed keywords with cache type
    if keywords:
        cache_data = {
            "keywords": keywords
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="keywords",
            ),
        )
    return keywords


async def mix_kg_vector_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    """
    Hybrid retrieval implementation combining knowledge graph and vector search.

    This function performs a hybrid search by:
    1. Extracting semantic information from knowledge graph
    2. Retrieving relevant text chunks through vector similarity
    3. Combining both results for comprehensive answer generation
    """
    # 1. Cache handling
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash("mix", query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, "mix", cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # 2. Execute knowledge graph and vector searches in parallel
    async def get_kg_context():
        try:
            # Extract keywords using extract_keywords_only function which already supports conversation history
            hl_keywords, ll_keywords = await extract_keywords_only(
                query, query_param, global_config, hashing_kv
            )

            if not hl_keywords and not ll_keywords:
                logger.warning(
                    "Both high-level and low-level keywords are empty")
                return None

            # Convert keyword lists to strings
            ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
            hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

            # Set query mode based on available keywords
            if not ll_keywords_str and not hl_keywords_str:
                return None
            elif not ll_keywords_str:
                query_param.mode = "global"
            elif not hl_keywords_str:
                query_param.mode = "local"
            else:
                query_param.mode = "hybrid"

            # Build knowledge graph context
            context = await _build_query_context(
                ll_keywords_str,
                hl_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )

            return context

        except Exception as e:
            logger.error(f"Error in get_kg_context: {str(e)}")
            return None

    async def get_vector_context():
        # Consider conversation history in vector search
        augmented_query = query
        if history_context:
            augmented_query = f"{history_context}\n{query}"

        try:
            # Reduce top_k for vector search in hybrid mode since we have structured information from KG
            mix_topk = min(10, query_param.top_k)
            # TODO: add ids to the query
            results = await chunks_vdb.query(
                augmented_query, top_k=mix_topk, ids=query_param.ids
            )
            if not results:
                return None

            chunks_ids = [r["id"] for r in results]
            chunks = await text_chunks_db.get_by_ids(chunks_ids)

            valid_chunks = []
            for chunk, result in zip(chunks, results):
                if chunk is not None and "content" in chunk:
                    # Merge chunk content and time metadata
                    chunk_with_time = {
                        "content": chunk["content"],
                        "created_at": result.get("created_at", None),
                    }
                    valid_chunks.append(chunk_with_time)

            if not valid_chunks:
                return None

            maybe_trun_chunks = truncate_list_by_token_size(
                valid_chunks,
                key=lambda x: x["content"],
                max_token_size=query_param.max_token_for_text_unit,
            )

            if not maybe_trun_chunks:
                return None

            # Include time information in content
            formatted_chunks = []
            for c in maybe_trun_chunks:
                chunk_text = c["content"]
                if c["created_at"]:
                    chunk_text = f"[Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c['created_at']))}]\n{chunk_text}"
                formatted_chunks.append(chunk_text)

            logger.debug(
                f"Truncate chunks from {len(chunks)} to {len(formatted_chunks)} (max tokens:{query_param.max_token_for_text_unit})"
            )
            return "\n--New Chunk--\n".join(formatted_chunks)
        except Exception as e:
            logger.error(f"Error in get_vector_context: {e}")
            return None

    # 3. Execute both retrievals in parallel
    kg_context, vector_context = await asyncio.gather(
        get_kg_context(), get_vector_context()
    )

    # 4. Merge contexts
    if kg_context is None and vector_context is None:
        return PROMPTS["fail_response"]

    if query_param.only_need_context:
        return {"kg_context": kg_context, "vector_context": vector_context}

    # 5. Construct hybrid prompt
    sys_prompt = (
        system_prompt
        if system_prompt
        else PROMPTS["mix_rag_response"].format(
            kg_context=kg_context
            if kg_context
            else "No relevant knowledge graph information found",
            vector_context=vector_context
            if vector_context
            else "No relevant text information found",
            response_type=query_param.response_type,
            history=history_context,
        )
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[mix_kg_vector_query]Prompt Tokens: {len_of_prompts}")

    # 6. Generate response
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    # Clean up response content
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        # 7. Save cache - Only cache after collecting complete response
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode="mix",
                cache_type="query",
            ),
        )

    return response


async def _build_query_context(
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    time_ranges: list
):
    logger.info(f"Process {os.getpid()} buidling query context...")
    if query_param.mode == "local":
        entities_context, relations_context, text_units_context, entity_relation_summary_context = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    elif query_param.mode == "local_plus":
        results_by_time = []
        for tr in time_ranges:
            print(f"🔍 查詢時間區間：{tr['start_date']} ~ {tr['end_date']}")

            entities_context, relations_context, text_units_context, entity_relation_summary_context = await _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
                tr  # ⬅️ 指定該段時間區間
            )

            # 跳過空結果
            if not entities_context.strip() and not relations_context.strip():
                continue

            results_by_time.append({
                "time_range": f"{tr['start_date']} ~ {tr['end_date']}",
                "entity_summary": entity_relation_summary_context,
                "text_units": text_units_context
            })
        # entities_context, relations_context, text_units_context, entity_relation_summary_context = await _get_node_data(
        #     ll_keywords,
        #     knowledge_graph_inst,
        #     entities_vdb,
        #     text_chunks_db,
        #     query_param,
        #     time_ranges[0]
        # )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context, entity_relation_summary_context = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    else:  # hybrid mode
        ll_data, hl_data = await asyncio.gather(
            _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            ),
            _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
            ),
        )

        (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        ) = ll_data

        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        ) = hl_data

        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    # not necessary to use LLM to generate a response
    if not entities_context.strip() and not relations_context.strip():
        return None
    # # MARK// 最終結果輸出
    # result = f"""
    # -----查詢實體與語意總結-----
    # {entity_relation_summary_context}
    # -----相關評論-----
    # {text_units_context}
    # """.strip()
    # # -----來源評論-----
    # print(result)
    # return result
    # 若全部時間區間都查無結果，回傳 None
    if not results_by_time:
        return None

    result_parts = []

    for r in results_by_time:
        section = f"""
    🕒【時間區間】：{r['time_range']}

    -----實體與語意總結-----
    {r['entity_summary']}

    -----相關評論-----
    {r['text_units']}
    """.strip()
        result_parts.append(section)

    # 用分隔線區分不同時間段的結果
    final_result = "\n\n==============================\n\n".join(result_parts)
    print(final_result)

    return final_result


async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    time_range: dict
):
    # get similar entities
    logger.info(
        f"Query nodes: {query}, top_k: {query_param.top_k}, cosine: {entities_vdb.cosine_better_than_threshold}"
    )

    results = await entities_vdb.query(
        query, top_k=query_param.top_k, ids=query_param.ids
    )

    print(f"_get_node_data: {time_range}")
    start_date = datetime.strptime(time_range['start_date'], "%Y-%m-%d").date()
    end_date = datetime.strptime(time_range['end_date'], "%Y-%m-%d").date()
    if not len(results):
        return "", "", ""
    # for r in results:
    #     print(f"{query} --> {r["entity_name"]} DISTANCE:{r["distance"]}")

    # get entity information
    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"], start_date, end_date) for r in results]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
        ),
    )

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk
    use_text_units, use_relations = await asyncio.gather(
        # _find_most_related_text_unit_from_entities(
        #     node_datas, query_param, text_chunks_db, knowledge_graph_inst
        # ),
        _find_incoming_chunk_texts_from_nodes(
            node_datas, query_param, text_chunks_db, knowledge_graph_inst, start_date, end_date
        ),
        _find_most_related_edges_from_entities(
            node_datas, query_param, knowledge_graph_inst, start_date, end_date
        ),
    )
    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_local_context,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} chunks"
    )

    # build prompt
    entites_section_list = [
        [
            "id",
            "實體名稱",
            "實體類型",
            "實體描述",
            "提及次數",
            "情感分數",
        ]
    ]
    for i, n in enumerate(node_datas):
        created_at = n.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n.get("mention_count", 0),
                n.get("sentiment_score", 0),
            ]
        )
    # entities_context = list_of_list_to_csv(entites_section_list)
    entity_descriptions = []
    for i, n in enumerate(node_datas):
        created_at = n.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        entity_descriptions.append(
            f"{i+1}. **{n['entity_name']}**（{n.get('entity_type', '未知類型')}）\n"
            f"　{n.get('description', '無描述')}根據評論資料， **{n['entity_name']}** 共被提及 **{n.get('mention_count', 0)} 次**，"
            f"情感分數為 **{n.get('sentiment_score', 0)}**\n"
        )

    entities_context = "\n".join(entity_descriptions)

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(use_relations):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                created_at,
            ]
        )

    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_descriptions = []
    for i, t in enumerate(use_text_units):
        text_units_descriptions.append(
            f"【{i+1}】來自「{t['entity_name']}」的段落：\n「{t['content']}」\n"
        )
    text_units_context = "\n".join(text_units_descriptions)
    entity_relation_summary_context = build_entity_relation_summary(
        use_relations)
    return entities_context, relations_context, text_units_context, entity_relation_summary_context


def build_entity_relation_summary(use_relations: list[dict]) -> str:
    from collections import defaultdict

    # 第一層聚合：來源 → [構面]
    graph = defaultdict(lambda: {
        "type": "UNKNOWN",
        "mention_count": 0,
        "sentiment_score": 0,
        "facets": defaultdict(lambda: {
            "mention_count": 0,
            "sentiment_score": 0,
        })
    })

    for rel in use_relations:
        src = rel.get("src_node")
        tgt = rel.get("tgt_node")
        src_degree = rel.get("rank")
        if not src or not tgt:
            continue

        src_name = src.get("entity_id")
        tgt_name = tgt.get("entity_id")

        # 來源資訊
        if graph[src_name]["mention_count"] == 0:
            graph[src_name]["type"] = src.get("entity_type", "未知類型")
            graph[src_name]["mention_count"] = src.get("mention_count", 0)
            graph[src_name]["sentiment_score"] = src.get("sentiment_score", 0)

        # 構面資訊
        graph[src_name]["facets"][tgt_name]["mention_count"] = tgt.get(
            "mention_count", 0)
        graph[src_name]["facets"][tgt_name]["sentiment_score"] = tgt.get(
            "sentiment_score", 0)

    # ⬇️ 組裝輸出
    output = []
    for src_name, data in graph.items():
        if (data['type'] == "構面" and data['mention_count'] > 0):
            output.append(
                f"🔸 **{src_name}**（{data['type']}）被提及了 {data['mention_count']} 次，情感比率 {float(data['sentiment_score']/data['mention_count'])}，其中："
            )
        elif (data['mention_count'] > 0):
            output.append(
                f"🔸 **{src_name}**（{data['type']}）被提及了 {data['mention_count']} 次，情感比率 {float(data['sentiment_score']/data['mention_count'])}，其中："
            )
        else:
            print(f"1992 就是{src_name}的提及次數 = {data['mention_count']}")
        for facet_name, f in data["facets"].items():
            # ⬇️ 如果 facet_name 是像「裕珍馨_商品」，就去掉「裕珍馨_」
            if facet_name.startswith(src_name + "_"):
                display_facet_name = facet_name[len(src_name) + 1:]
            else:
                display_facet_name = facet_name
            if (f['mention_count'] > 0):
                output.append(
                    f"  - {display_facet_name}（{f['mention_count']} 次 / {float(f['sentiment_score']/f['mention_count'])}）"
                )
        output.append("")  # 空行區隔

    return "\n".join(output)


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(
            v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    tasks = []

    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = index
                tasks.append((c_id, index, this_edges))

    results = await asyncio.gather(
        *[text_chunks_db.get_by_id(c_id) for c_id, _, _ in tasks]
    )

    for (c_id, index, this_edges), data in zip(tasks, results):
        all_text_units_lookup[c_id] = {
            "data": data,
            "order": index,
            "relation_counts": 0,
        }

        if this_edges:
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    logger.debug(
        f"Truncate chunks from {len(all_text_units_lookup)} to {len(all_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    # all_text_units = [t["data"] for t in all_text_units]
    # return all_text_units
    return [
        {
            "id": t["id"],
            "content": t["data"]["content"],
            "entity_name": node_datas[t["order"]]["entity_name"]
        }
        for t in all_text_units
    ]


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    start_date: datetime.date,
    end_date: datetime.date
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"], start_date, end_date) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for sorted_edge in this_edges:
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    # 🟡 同步查詢邊、source節點、target節點
    all_edges_pack, src_nodes, tgt_nodes, node_degree = await asyncio.gather(
        asyncio.gather(*[knowledge_graph_inst.get_edge(e[0], e[1])
                       for e in all_edges]),
        asyncio.gather(*[knowledge_graph_inst.get_node(e[0], start_date, end_date)
                       for e in all_edges]),
        asyncio.gather(*[knowledge_graph_inst.get_node(e[1], start_date, end_date, root_id=e[0])
                       for e in all_edges]),  # 🆕 新增查 target node
        asyncio.gather(
            *[knowledge_graph_inst.node_degree(e[0]) for e in all_edges]
        ),
    )

    # 🔵 整合邊與 source/target 節點資訊
    all_edges_data = []
    for k, v, src_node, tgt_node, d in zip(all_edges, all_edges_pack, src_nodes, tgt_nodes, node_degree):
        if v is None:
            continue
        all_edges_data.append({
            "src_tgt": k,
            "rank": d,
            "src_node": src_node,  # 🆕 可選：附上來源節點資料
            "tgt_node": tgt_node,  # 🆕 附上目標節點（鄰居）資料
            **v
        })
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_global_context,
    )
    logger.debug(
        f"Truncate relations from {len(all_edges)} to {len(all_edges_data)} (max tokens:{query_param.max_token_for_global_context})"
    )
    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords}, top_k: {query_param.top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )

    results = await relationships_vdb.query(
        keywords, top_k=query_param.top_k, ids=query_param.ids
    )

    if not len(results):
        return "", "", ""

   # 取得邊資料
    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [e for e in edge_datas if e is not None and e.get(
        "source_id") is not None]

    # 取得來源節點的 mention_count
    src_nodes = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["src_id"]) for r in results]
    )

    edge_datas = [
        {
            "src_id": k["src_id"],
            "tgt_id": k["tgt_id"],
            "rank": src_node.get("mention_count", 0) if src_node else 0,
            "created_at": k.get("__created_at__", None),
            **v,
        }
        for k, v, src_node in zip(results, edge_datas, src_nodes)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_global_context,
    )
    use_entities, use_text_units = await asyncio.gather(
        _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        ),
        _find_related_text_unit_from_relationships(
            edge_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} chunks"
    )

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(edge_datas):
        created_at = e.get("created_at", "Unknown")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "實體名稱", "實體類型", "實體描述", "提及次數", "情感分數"]]
    for i, n in enumerate(use_entities):
        created_at = e.get("created_at", "Unknown")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n.get("mention_count", 0),
                n.get("sentiment_score", 0)
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[
                knowledge_graph_inst.get_node(entity_name)
                for entity_name in entity_names
            ]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.node_degree(entity_name)
                for entity_name in entity_names
            ]
        ),
    )
    node_datas = [e for e in node_datas if e and e.get("source_id")]
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_local_context,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    return node_datas


async def _find_incoming_chunk_texts_from_nodes(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    start_date: datetime.date,
    end_date: datetime.date
):
    edge_datas = []

    for node in node_datas:
        entity_id = node["entity_id"]

        src_node = await knowledge_graph_inst.get_node(entity_id, start_date, end_date)

        if not src_node:
            continue

        # ✅ 改這裡：從 source_ids 取出多筆 chunk
        if "source_ids" in src_node and isinstance(src_node["source_ids"], list):
            for chunk_id in src_node["source_ids"]:
                edge_datas.append(
                    {"entity_name": entity_id, "source_id": chunk_id})

    return await _find_related_text_unit_from_relationships(
        edge_datas=edge_datas,
        query_param=query_param,
        text_chunks_db=text_chunks_db,
        knowledge_graph_inst=knowledge_graph_inst
    )


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    # 🔧 建立一個 chunk_id → entity_name 的對應表
    chunk_to_entity = {}
    for dp in edge_datas:
        entity_name = dp["entity_name"]
        chunk_ids = split_string_by_multi_markers(
            dp["source_id"], [GRAPH_FIELD_SEP])
        for c_id in chunk_ids:
            chunk_to_entity[c_id] = entity_name

    async def fetch_chunk_data(c_id, index):
        if c_id not in all_text_units_lookup:
            chunk_data = await text_chunks_db.get_by_id(c_id)
            # Only store valid data
            if chunk_data is not None and "content" in chunk_data:
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                    "entity_name": chunk_to_entity.get(c_id, None),
                }

    tasks = []
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            tasks.append(fetch_chunk_data(c_id, index))

    await asyncio.gather(*tasks)

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    logger.debug(
        f"Truncate chunks from {len(valid_text_units)} to {len(truncated_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units: list[TextChunkSchema] = [
        {
            **t["data"],
            "entity_name": t.get("entity_name")  # 🆕 最後加進去
        }
        for t in truncated_text_units
    ]
    print(all_text_units)

    return all_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    results = await chunks_vdb.query(
        query, top_k=query_param.top_k, ids=query_param.ids
    )
    if not len(results):
        return PROMPTS["fail_response"]

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return PROMPTS["fail_response"]

    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return PROMPTS["fail_response"]

    logger.debug(
        f"Truncate chunks from {len(chunks)} to {len(maybe_trun_chunks)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    section = "\n--New Chunk--\n".join([c["content"]
                                       for c in maybe_trun_chunks])

    if query_param.only_need_context:
        return section

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section,
        response_type=query_param.response_type,
        history=history_context,
    )
    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[naive_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt):]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )

    return response


async def kg_query_with_keywords(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> str | AsyncIterator[str]:
    """
    Refactored kg_query that does NOT extract keywords by itself.
    It expects hl_keywords and ll_keywords to be set in query_param, or defaults to empty.
    Then it uses those to build context and produce a final LLM response.
    """

    # ---------------------------
    # 1) Handle potential cache for query results
    # ---------------------------
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # ---------------------------
    # 2) RETRIEVE KEYWORDS FROM query_param
    # ---------------------------

    # If these fields don't exist, default to empty lists/strings.
    hl_keywords = getattr(query_param, "hl_keywords", []) or []
    ll_keywords = getattr(query_param, "ll_keywords", []) or []

    # If neither has any keywords, you could handle that logic here.
    if not hl_keywords and not ll_keywords:
        logger.warning(
            "No keywords found in query_param. Could default to global mode or fail."
        )
        return PROMPTS["fail_response"]
    if not ll_keywords and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            "low_level_keywords is empty, switching to global mode.")
        query_param.mode = "global"
    if not hl_keywords and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            "high_level_keywords is empty, switching to local mode.")
        query_param.mode = "local"

    # Flatten low-level and high-level keywords if needed
    ll_keywords_flat = (
        [item for sublist in ll_keywords for item in sublist]
        if any(isinstance(i, list) for i in ll_keywords)
        else ll_keywords
    )
    hl_keywords_flat = (
        [item for sublist in hl_keywords for item in sublist]
        if any(isinstance(i, list) for i in hl_keywords)
        else hl_keywords
    )

    # Join the flattened lists
    ll_keywords_str = ", ".join(ll_keywords_flat) if ll_keywords_flat else ""
    hl_keywords_str = ", ".join(hl_keywords_flat) if hl_keywords_flat else ""

    # ---------------------------
    # 3) BUILD CONTEXT
    # ---------------------------
    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )
    if not context:
        return PROMPTS["fail_response"]

    # If only context is needed, return it
    if query_param.only_need_context:
        return context

    # ---------------------------
    # 4) BUILD THE SYSTEM PROMPT + CALL LLM
    # ---------------------------

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )
    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query_with_keywords]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    # 清理响应内容
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        # 7. Save cache - 只有在收集完整响应后才缓存
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                cache_type="query",
            ),
        )

    return response


async def query_with_keywords(
    query: str,
    prompt: str,
    param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> str | AsyncIterator[str]:
    """
    Extract keywords from the query and then use them for retrieving information.

    1. Extracts high-level and low-level keywords from the query
    2. Formats the query with the extracted keywords and prompt
    3. Uses the appropriate query method based on param.mode

    Args:
        query: The user's query
        prompt: Additional prompt to prepend to the query
        param: Query parameters
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entities vector database
        relationships_vdb: Relationships vector database
        chunks_vdb: Document chunks vector database
        text_chunks_db: Text chunks storage
        global_config: Global configuration
        hashing_kv: Cache storage

    Returns:
        Query response or async iterator
    """
    # Extract keywords
    hl_keywords, ll_keywords = await extract_keywords_only(
        text=query,
        param=param,
        global_config=global_config,
        hashing_kv=hashing_kv,
    )

    param.hl_keywords = hl_keywords
    param.ll_keywords = ll_keywords

    # Create a new string with the prompt and the keywords
    ll_keywords_str = ", ".join(ll_keywords)
    hl_keywords_str = ", ".join(hl_keywords)
    formatted_question = f"{prompt}\n\n### Keywords:\nHigh-level: {hl_keywords_str}\nLow-level: {ll_keywords_str}\n\n### Query:\n{query}"

    # Use appropriate query method based on mode
    if param.mode in ["local", "global", "hybrid"]:
        return await kg_query_with_keywords(
            formatted_question,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    elif param.mode == "naive":
        return await naive_query(
            formatted_question,
            chunks_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    elif param.mode == "mix":
        return await mix_kg_vector_query(
            formatted_question,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            chunks_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    else:
        raise ValueError(f"Unknown mode {param.mode}")
