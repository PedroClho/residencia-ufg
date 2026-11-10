"""Agentes de contexto longo, BM25 e embedding copiados de `agent.py` (MemoryAgentBench, MIT).

Origem: commit 538026089d1a8a8eff05121d0db89b388f360eba, `agent.py` e
`methods/embedding_retriever.py` (`RAGSystem.answer_query`, `TextRetriever.retrieve`). Nomes e
corpos são os do autor; as únicas mudanças estão listadas em `NOTICE.md` desta pasta.
"""
import re
import time
from typing import Dict, List, Union

import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi

from replica.autor.templates import get_template


class AgentWrapper:
    """
    A wrapper class for different types of memory agents including:
    - Long context agents (GPT, Claude, Gemini)
    - RAG agents (BM25, text-embedding-3-small)
    """

    def __init__(self, agent_config, dataset_config, cliente, embeddings=None, obter_template=get_template):
        """
        Initialize the agent wrapper with specified configuration.

        Args:
            agent_config: Configuration dictionary for the agent
            dataset_config: Configuration dictionary for the dataset
            cliente: ClienteOpenAI ou ClienteAnthropic (replica.clientes), com o método `conversar`
            embeddings: ClienteEmbeddings (replica.clientes), só para o agente de embedding
            obter_template: função com a assinatura de `get_template` (a OAB pluga a sua)
        """
        # Basic agent configuration
        self.agent_name = agent_config['agent_name']
        self.sub_dataset = dataset_config['sub_dataset']
        self.context_max_length = dataset_config['context_max_length']
        self.dataset = dataset_config['dataset']

        # Output and storage configuration
        self.output_dir = agent_config['output_dir']

        # Context and token limits
        self.input_length_limit = (agent_config['input_length_limit'] -
                                 agent_config['buffer_length'] -
                                 dataset_config['generation_max_length'])

        # Model configuration
        self.model = agent_config['model']
        self.max_tokens = dataset_config['generation_max_length']
        self.temperature = agent_config.get('temperature', 0.0)

        # Initialize tokenizer (default to gpt-4o-mini for non-gpt models)
        model_for_tokenizer = self.model if "gpt-4o" in self.model else "gpt-4o-mini"
        self.tokenizer = tiktoken.encoding_for_model(model_for_tokenizer)

        # Clientes com teto de gasto e despacho de templates (adaptações 1, 4 e 6)
        self.cliente = cliente
        self.embeddings = embeddings
        self.obter_template = obter_template

        # Initialize agent based on type
        self._initialize_agent_by_type(agent_config, dataset_config)

    def _initialize_agent_by_type(self, agent_config, dataset_config):
        """Initialize the specific agent type based on agent name."""

        if 'Long_context_agent' in self.agent_name:
            self._initialize_long_context_agent()
        elif self._is_agent_type("rag"):
            self._initialize_rag_agent(agent_config, dataset_config)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _is_agent_type(self, agent_type):
        """Check if the current agent is of a specific type."""
        return agent_type in self.agent_name

    def _create_standard_response(self, output, input_tokens, output_tokens, memory_time, query_time):
        """Create standardized response dictionary."""
        return {
            "output": output,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
        }

    def _anexar_contagem(self, saida, resposta):
        """Acrescenta ao dicionário do autor a contagem e o custo do livro-caixa (`Resposta`)."""
        saida.update(
            tokens_entrada=resposta.tokens_entrada,
            tokens_saida=resposta.tokens_saida,
            tokens_cacheados=resposta.tokens_cacheados,
            tokens_escritos_cache=resposta.tokens_escritos_cache,
            custo_usd=resposta.custo_usd,
            erro=resposta.erro,
        )
        return saida

    def _initialize_long_context_agent(self):
        """Initialize long context agent with appropriate client."""
        self.context = ''

    def _initialize_rag_agent(self, agent_config, dataset_config):
        """Initialize RAG agent with retrieval configuration."""
        self.context = ''
        self.chunks = []
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = dataset_config['chunk_size']
        self.context_len = 0
        self.context_id = -1

    def send_message(self, message, memorizing=False, query_id=None, context_id=None):
        """
        Send a message to the agent for either memorization or querying.

        Args:
            message: The message content (context for memorization, query for answering)
            memorizing: Whether to memorize the message (True) or answer it (False)
            query_id: Unique identifier for the query
            context_id: Unique identifier for the context

        Returns:
            dict or str: Agent response with metadata (for queries) or confirmation (for memorization)
        """
        # Route to appropriate agent handler based on agent type
        if 'Long_context_agent' in self.agent_name:
            return self._handle_long_context_agent(message, memorizing)
        elif self._is_agent_type("rag"):
            return self._handle_rag_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _handle_long_context_agent(self, message, memorizing):
        """Handle message processing for long context agents."""
        if memorizing:
            # Add message to context memory
            memorize_template = self.obter_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:
            # Process query with context
            return self._query_long_context_agent(message)

    def _query_long_context_agent(self, message):
        """Process a query for long context agents."""
        # Get appropriate tokenizer
        try:
            tokenizer = tiktoken.encoding_for_model(self.model)
        except:
            tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

        # Handle context truncation for non-long context models
        buffer_length = 50000
        if self.input_length_limit <= self.context_max_length + buffer_length:
            self._truncate_context_if_needed(tokenizer)

        # Format message with context and system prompt
        # O texto enviado é o full_message do autor: (self.context + "\n") + message, com o
        # contexto como prefixo cacheável e a pergunta depois dele.
        system_message = self.obter_template(self.sub_dataset, 'system', self.agent_name)

        # Query the model
        start_time = time.time()

        resposta = self.cliente.conversar(
            system_message,
            message,
            max_tokens=self.max_tokens,
            prefixo_cacheavel=self.context + "\n",
        )
        return self._anexar_contagem(
            self._create_standard_response(
                resposta.texto,
                resposta.tokens_entrada,
                resposta.tokens_saida,
                0,
                time.time() - start_time
            ),
            resposta,
        )

    def _truncate_context_if_needed(self, tokenizer):
        """Truncate context if it exceeds limits."""
        # Truncate context if it exceeds the context_max_length
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.context_max_length:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.context_max_length:])

        # Truncate if context exceeds the input_length_limit
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])

    def _handle_rag_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for RAG agents."""
        if memorizing:
            # Add message to chunks and context
            memorize_template = self.obter_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            self.chunks.append(formatted_message)
            self.context_len = self.context_len + self.chunk_size

            # Truncate context if it exceeds limits
            if self.context_len > self.input_length_limit:
                self.chunks = self.chunks[1:]
                self.context_len = self.context_len - self.chunk_size
            return ''
        else:
            # Handle query processing for different RAG types
            return self._process_rag_query(message, query_id, context_id)

    def _process_rag_query(self, message, query_id, context_id):
        """Process query for RAG agents with different retrieval strategies."""

        # Truncate context if needed
        tokenizer = self.tokenizer
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])
        if self.context_len > self.input_length_limit:
            self.chunks = self.chunks[1:]
            self.context_len = self.context_len - self.chunk_size

        # Route to specific RAG implementation and get result
        rag_handlers = {
            "rag_bm25": lambda: self._handle_bm25_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_small": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
        }

        # Find matching handler
        handler = next((handler for agent_type, handler in rag_handlers.items() if self._is_agent_type(agent_type)), None)
        if not handler:
            raise NotImplementedError(f"RAG agent type not supported: {self.agent_name}")

        output = handler()

        return output

    # RAG implementation methods
    def _handle_bm25_rag(self, message, context_id, tokenizer):
        """Handle BM25 RAG processing."""
        start_time = time.time()

        # Extract retrieval query from message
        retrieval_query = self._extract_retrieval_query(message)

        # Build vectorstore if context changed
        if self.context_id != context_id:
            self.bm25_retriever = BM25Okapi([c.split() for c in self.chunks])

        # Retrieve documents
        bm25_documents = self.bm25_retriever.get_top_n(retrieval_query.split(), self.chunks, n=self.retrieve_num)
        retrieval_context = [f"{doc}\n" for doc in bm25_documents]
        memory_construction_time = time.time() - start_time

        # Answer the query
        retrieval_memory_string = "\n".join([f"Memory {i+1}:\n{text}" for i, text in enumerate(retrieval_context)])

        # Format the message
        ask_llm_message = retrieval_memory_string + "\n" + message
        system_message = self.obter_template(self.sub_dataset, 'system', self.agent_name)

        # Generate response
        resposta = self.cliente.conversar(
            system_message,
            ask_llm_message,
            max_tokens=self.max_tokens,
            prefixo_cacheavel=None,
        )

        query_time_len = time.time() - start_time - memory_construction_time
        self.context_id = context_id

        return self._anexar_contagem({
            "output": resposta.texto,
            "input_len": resposta.tokens_entrada,
            "output_len": resposta.tokens_saida,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }, resposta)

    def _extract_retrieval_query(self, message):
        """Extract retrieval query from message using regex patterns."""
        patterns = [
            r"Now Answer the Question:\s*(.*)",
            r"Here is the conversation:\s*(.*)",
            r"Agora responda à questão:\s*(.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.DOTALL)
            if match:
                return ''.join(match.groups())

        return message

    def _handle_embedding_rag(self, message, context_id, tokenizer):
        """Handle embedding-based RAG processing (Contriever, Text-embedding models)."""
        # Build vectorstore if context changed
        if self.context_id != context_id:
            vetores = np.asarray(self.embeddings.embutir(self.chunks), dtype=np.float32)
            self.vetores_chunks = vetores / np.linalg.norm(vetores, axis=1, keepdims=True)

        # Retrieve relevant passages and answer the query
        system_message = self.obter_template(self.sub_dataset, 'system', self.agent_name)
        result = self.answer_query(
            query=message,
            top_k=self.retrieve_num,
            system_message=system_message
        )
        retrieval_context = result['context_used']

        self.context_id = context_id

        return self._anexar_contagem({
            "output": result["answer"],
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(result["answer"], disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }, result["resposta"])

    # Copiado de methods/embedding_retriever.py (TextRetriever.retrieve e RAGSystem.answer_query)
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant contexts for a query (auto-caches vectorstore)

        Args:
            query: The search query
            top_k: Number of documents to retrieve from vector store
        """
        initial_k = top_k

        # Perform similarity search to get initial results
        # (FAISS L2 sobre vetores normalizados ordena igual ao produto interno)
        consulta = np.asarray(self.embeddings.embutir([query]), dtype=np.float32)[0]
        consulta = consulta / np.linalg.norm(consulta)
        pontuacoes = self.vetores_chunks @ consulta
        results = np.argsort(-pontuacoes, kind="stable")[:initial_k]
        retrieved_docs = [self.chunks[i] for i in results]

        # Return results (truncated to top_k if needed)
        return retrieved_docs[:top_k]

    def answer_query(self, query: str, top_k: int, system_message: str) -> Dict[str, Union[str, float]]:
        """Retrieve relevant information and generate an answer"""
        # Retrieve relevant passages
        start_time = time.time()
        retrieval_query = self._extract_retrieval_query(query)
        retrieved_contexts = self.retrieve(retrieval_query, top_k)

        # Format retrieved contexts
        formatted_context = "\n\n".join([f"Passage {i+1}:\n{text}"
                                       for i, text in enumerate(retrieved_contexts)])
        memory_construction_time = time.time() - start_time

        # Generate prompt
        retrieval_memory_string = "\n".join([f"Memory {i+1}:\n{text}" for i, text in enumerate(retrieved_contexts)])
        ask_llm_message=retrieval_memory_string + "\n" + query

        # Get response from LLM
        resposta = self.cliente.conversar(
            system_message,
            ask_llm_message,
            max_tokens=self.max_tokens,
            prefixo_cacheavel=None,
        )
        query_time_len = time.time() - start_time - memory_construction_time

        return {
            "query": query,
            "context_used": formatted_context,
            "answer": resposta.texto,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "resposta": resposta,
        }
