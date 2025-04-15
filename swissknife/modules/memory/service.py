import os
import chromadb
import datetime
import uuid
import json
from typing import List, Dict, Any


class MemoryService:
    """Service for storing and retrieving conversation memory using ChromaDB."""

    def __init__(self, collection_name="conversation", persist_directory=None):
        """
        Initialize the memory service with ChromaDB.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the ChromaDB data
        """
        # Ensure the persist directory exists
        self.db_path = os.getenv("MEMORYDB_PATH", "./memory_db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Create or get collection for storing memories
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.message_raw_collection = self.client.get_or_create_collection(
            name="message_raw"
        )

        # Configuration for chunking
        self.chunk_size = 200  # words per chunk
        self.chunk_overlap = 40  # words overlap between chunks

    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: The text to split into chunks

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            return [text]

        i = 0
        while i < len(words):
            chunk_end = min(i + self.chunk_size, len(words))
            chunk = " ".join(words[i:chunk_end])
            chunks.append(chunk)
            i += self.chunk_size - self.chunk_overlap

        return chunks

    def store_conversation(
        self, user_message: str, assistant_response: str
    ) -> List[str]:
        """
        Store a conversation exchange in memory.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response

        Returns:
            List of memory IDs created
        """
        # Create the memory document by combining user message and response
        conversation_text = f"Date: {datetime.datetime.today().strftime('%Y-%m-%d')}.\n\n User: {user_message}.\n\nAssistant: {assistant_response}"

        # Split into chunks
        chunks = self._create_chunks(conversation_text)

        # Store each chunk with metadata
        memory_ids = []
        timestamp = datetime.datetime.now().isoformat()

        for i, chunk in enumerate(chunks):
            memory_id = str(uuid.uuid4())
            memory_ids.append(memory_id)

            # Add to ChromaDB collection
            self.collection.add(
                documents=[chunk],
                metadatas=[
                    {
                        "timestamp": timestamp,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "conversation_id": memory_ids[
                            0
                        ],  # First ID is the conversation ID
                        "type": "conversation",
                    }
                ],
                ids=[memory_id],
            )

        return memory_ids

    def store_message_markdown(
        self, message: Dict[str, Any], conversation_id: str
    ) -> str:
        """
        Store a raw message (user or assistant) as a markdown document in the memory database.

        Args:
            message: The message dictionary to store.
            conversation_id: The conversation ID to associate with this message.

        Returns:
            The memory ID created.
        """
        # Standardize the message format
        std_msg = {"role": message.get("role", "")}
        std_msg["agent"] = message.get("agent", "")

        # Handle content
        if "content" in message:
            if (
                isinstance(message["content"], str)
                and message.get("role", "") == "assistant"
            ):
                std_msg["content"] = [{"type": "text", "text": message["content"]}]

            else:
                std_msg["content"] = message["content"]

        # Handle tool calls
        if "tool_calls" in message:
            std_msg["tool_calls"] = []
            for tool_call in message["tool_calls"]:
                std_tool_call = {
                    "id": tool_call.get("id"),
                    "name": tool_call.get("function", {}).get("name"),
                    "arguments": json.loads(
                        tool_call.get("function", {}).get("arguments")
                    ),
                    "type": tool_call.get("type", "function"),
                }
                std_msg["tool_calls"].append(std_tool_call)

        # Handle tool results
        if message.get("role") == "tool":
            std_msg["tool_result"] = {
                "tool_use_id": message.get("tool_call_id"),
                "content": message.get("content"),
                "is_error": message.get("content", "").startswith("ERROR:"),
            }

        # Format as a markdown document
        md_lines = [f"**Role:** {std_msg['role']}  "]
        if std_msg["role"] == "assistant":
            md_lines.append(f"**Agent:** {std_msg.get('agent', '')}  ")

        if "content" in std_msg:
            md_lines.append(f"**Content:**\n\n{std_msg['content']}\n")

        if "tool_calls" in std_msg:
            md_lines.append("**Tool Calls:**")
            for tool_call in std_msg["tool_calls"]:
                md_lines.append(f"- **ID:** {tool_call.get('id', '')}")
                md_lines.append(f"  - **Name:** {tool_call.get('name', '')}")
                md_lines.append(f"  - **Type:** {tool_call.get('type', '')}")
                md_lines.append(f"  - **Arguments:**\n")
                md_lines.append(
                    f"    ```json\n{json.dumps(tool_call.get('arguments', {}), indent=2, ensure_ascii=False)}\n    ```"
                )

        if "tool_result" in std_msg:
            tool_result = std_msg["tool_result"]
            md_lines.append("**Tool Result:**")
            md_lines.append(f"- **Tool Use ID:** {tool_result.get('tool_use_id', '')}")
            md_lines.append(f"- **Content:**\n\n{tool_result.get('content', '')}")
            md_lines.append(f"- **Is Error:** {tool_result.get('is_error', False)}")

        markdown_doc = "\n".join(md_lines)

        memory_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()

        self.message_raw_collection.add(
            documents=[markdown_doc],
            metadatas=[
                {
                    "timestamp": timestamp,
                    "conversation_id": conversation_id,
                    "type": "message_markdown",
                }
            ],
            ids=[memory_id],
        )
        return memory_id

    def retrieve_memory(self, keywords: str, limit: int = 5) -> str:
        """
        Retrieve relevant memories based on keywords.

        Args:
            keywords: Keywords to search for
            limit: Maximum number of results to return

        Returns:
            Formatted string of relevant memories
        """
        results = self.collection.query(query_texts=[keywords], n_results=limit)

        if not results["documents"] or not results["documents"][0]:
            return "No relevant memories found."

        # Group chunks by conversation_id
        conversation_chunks = {}
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            conv_id = metadata.get("conversation_id", "unknown")
            if conv_id not in conversation_chunks:
                conversation_chunks[conv_id] = {
                    "chunks": [],
                    "timestamp": metadata.get("timestamp", "unknown"),
                    "relevance": len(results["documents"][0])
                    - i,  # Higher relevance for earlier results
                }
            conversation_chunks[conv_id]["chunks"].append(
                (metadata.get("chunk_index", 0), doc)
            )

        # Sort conversations by relevance
        sorted_conversations = sorted(
            conversation_chunks.items(), key=lambda x: x[1]["relevance"], reverse=True
        )

        # Format the output
        output = []
        for conv_id, conv_data in sorted_conversations:
            # Sort chunks by index
            sorted_chunks = sorted(conv_data["chunks"], key=lambda x: x[0])
            conversation_text = "\n".join([chunk for _, chunk in sorted_chunks])

            # Format timestamp
            timestamp = "Unknown time"
            if conv_data["timestamp"] != "unknown":
                try:
                    dt = datetime.datetime.fromisoformat(conv_data["timestamp"])
                    timestamp = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    timestamp = conv_data["timestamp"]

            output.append(f"--- Memory from {timestamp} ---\n{conversation_text}")

        return "\n\n".join(output)

    def cleanup_old_memories(self, months: int = 1) -> int:
        """
        Remove memories older than the specified number of months.

        Args:
            months: Number of months to keep

        Returns:
            Number of memories removed
        """
        # Calculate the cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30 * months)

        # Get all memories
        all_memories = self.collection.get()

        # Find IDs to remove
        ids_to_remove = []
        for i, metadata in enumerate(all_memories["metadatas"]):
            # Parse timestamp string to datetime object for proper comparison
            timestamp_str = metadata.get(
                "timestamp", datetime.datetime.now().isoformat()
            )
            try:
                timestamp_dt = datetime.datetime.fromisoformat(timestamp_str)
                if timestamp_dt < cutoff_date:
                    ids_to_remove.append(all_memories["ids"][i])
            except ValueError:
                # If timestamp can't be parsed, consider it as old and remove it
                ids_to_remove.append(all_memories["ids"][i])
                ids_to_remove.append(all_memories["ids"][i])

        # Remove the old memories
        if ids_to_remove:
            self.collection.delete(ids=ids_to_remove)

        return len(ids_to_remove)

    def forget_topic(self, topic: str) -> Dict[str, Any]:
        """
        Remove memories related to a specific topic based on keyword search.

        Args:
            topic: Keywords describing the topic to forget

        Returns:
            Dict with success status and information about the operation
        """
        try:
            # Query for memories related to the topic
            results = self.collection.query(query_texts=[topic], n_results=100)

            if not results["documents"] or not results["documents"][0]:
                return {
                    "success": False,
                    "message": f"No memories found related to '{topic}'",
                    "count": 0,
                }

            # Collect all conversation IDs related to the topic
            conversation_ids = set()
            for metadata in results["metadatas"][0]:
                conv_id = metadata.get("conversation_id")
                if conv_id:
                    conversation_ids.add(conv_id)

            # Get all memories to find those with matching conversation IDs
            all_memories = self.collection.get()

            # Find IDs to remove
            ids_to_remove = []
            for i, metadata in enumerate(all_memories["metadatas"]):
                if metadata.get("conversation_id") in conversation_ids:
                    ids_to_remove.append(all_memories["ids"][i])

            # Remove the memories
            if ids_to_remove:
                self.collection.delete(ids=ids_to_remove)

            return {
                "success": True,
                "message": f"Successfully removed {len(ids_to_remove)} memory chunks related to '{topic}'",
                "count": len(ids_to_remove),
                "conversations_affected": len(conversation_ids),
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error forgetting topic: {str(e)}",
                "count": 0,
            }

    def retrieve_message_markdown(
        self, keywords: List[str], conversation_id: str
    ) -> str:  # Changed return type hint to str
        """
        Retrieve up to 3 most relevant markdown messages from message_raw_collection
        for a given conversation_id, filtered by keywords, and return them as a
        single markdown formatted string.

        Args:
            conversation_id: The conversation ID to filter messages.
            keywords: List of keywords to search for relevance.

        Returns:
            A single markdown string containing the retrieved messages, separated by
            a horizontal rule, or a message indicating none were found.
        """
        # Query the message_raw_collection
        results = self.message_raw_collection.query(
            query_texts=keywords,  # Use the joined query string
            n_results=3,  # Directly request only 3 results
            where={"conversation_id": conversation_id},
        )

        if not results["documents"] or not results["documents"][0]:
            return "No relevant messages found for this conversation and keywords."

        # Extract documents if available
        docs = results.get("documents", [[]])[0]

        # Format the output as a single markdown string
        if not docs:
            return "No relevant messages found for this conversation and keywords."
        else:
            # Join the markdown documents with a separator
            return "\n\n---\n\n".join(docs)
