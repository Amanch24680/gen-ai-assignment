document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const queryInput = document.getElementById("query-input");
    const sendButton = document.getElementById("send-button");
    const chatMessages = document.getElementById("chat-messages");
    const emptyState = document.getElementById("empty-state");
    const loadingIndicator = document.getElementById("loading-indicator");
    const errorContainer = document.getElementById("error-container");
    const errorMessage = document.getElementById("error-message");
    const errorCloseBtn = document.getElementById("error-close-btn");
    const statusIndicator = document.getElementById("status-indicator");
    const statusText = document.getElementById("status-text");

    let isSubmitting = false;
    const API_ENDPOINT = "/api/v1/query";

    // Handle Keyboard Shortcuts (Enter sends, Shift+Enter new line)
    queryInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            chatForm.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
        }
    });

    // Handle Sample Query Chips Click
    document.querySelectorAll(".sample-chip").forEach((chip) => {
        chip.addEventListener("click", () => {
            const query = chip.getAttribute("data-query");
            if (query) {
                queryInput.value = query;
                chatForm.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
            }
        });
    });

    // Handle Form Submission
    chatForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const queryText = queryInput.value.trim();

        if (!queryText || isSubmitting) {
            return;
        }

        // Hide empty state on first query
        if (emptyState) {
            emptyState.style.display = "none";
        }

        // Hide any previous errors
        hideError();

        // Append User Message to UI
        appendUserMessage(queryText);

        // Clear input field & set loading state
        queryInput.value = "";
        setLoadingState(true);

        try {
            const response = await fetch(API_ENDPOINT, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ query: queryText }),
            });

            if (!response.ok) {
                let errorDetail = `HTTP ${response.status} Error`;
                try {
                    const errorJson = await response.json();
                    if (errorJson && errorJson.detail) {
                        errorDetail = errorJson.detail;
                    }
                } catch (e) {
                    // Ignore JSON parsing failure for non-JSON error
                }
                throw new Error(errorDetail);
            }

            const data = await response.json();
            appendAssistantMessage(data);

        } catch (error) {
            showError(`Failed to process query: ${error.message}`);
        } finally {
            setLoadingState(false);
            queryInput.focus();
        }
    });

    // Close Error Banner
    if (errorCloseBtn) {
        errorCloseBtn.addEventListener("click", () => {
            hideError();
        });
    }

    // Helper: Set UI Loading State
    function setLoadingState(loading) {
        isSubmitting = loading;
        sendButton.disabled = loading;
        queryInput.disabled = loading;

        if (loading) {
            loadingIndicator.style.display = "flex";
            if (statusIndicator) {
                statusIndicator.classList.add("busy");
            }
            if (statusText) {
                statusText.textContent = "Processing";
            }
        } else {
            loadingIndicator.style.display = "none";
            if (statusIndicator) {
                statusIndicator.classList.remove("busy");
            }
            if (statusText) {
                statusText.textContent = "Ready";
            }
        }
    }

    // Helper: Show Error Banner
    function showError(message) {
        errorMessage.textContent = message;
        errorContainer.style.display = "flex";
    }

    // Helper: Hide Error Banner
    function hideError() {
        errorContainer.style.display = "none";
        errorMessage.textContent = "";
    }

    // Helper: Append User Message to Chat Window
    function appendUserMessage(text) {
        const messageCard = document.createElement("div");
        messageCard.className = "message-card message-user";

        const senderLabel = document.createElement("div");
        senderLabel.className = "message-sender";
        senderLabel.textContent = "You";

        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        contentDiv.textContent = text;

        messageCard.appendChild(senderLabel);
        messageCard.appendChild(contentDiv);

        chatMessages.appendChild(messageCard);
        scrollToBottom();
    }

    // Helper: Append Assistant Answer to Chat Window
    function appendAssistantMessage(data) {
        const messageCard = document.createElement("div");
        messageCard.className = "message-card message-assistant";

        const senderLabel = document.createElement("div");
        senderLabel.className = "message-sender";
        senderLabel.textContent = "Assistant";

        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        contentDiv.textContent = data.answer || "No answer generated.";

        messageCard.appendChild(senderLabel);
        messageCard.appendChild(contentDiv);

        // Render Citations if available
        if (data.citations && data.citations.length > 0) {
            const citationsContainer = document.createElement("div");
            citationsContainer.className = "citations-container";

            const citationsTitle = document.createElement("div");
            citationsTitle.className = "citations-title";
            citationsTitle.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                <span>Retrieved Sources (${data.citations.length})</span>
            `;
            citationsContainer.appendChild(citationsTitle);

            const citationsList = document.createElement("div");
            citationsList.className = "citations-list";

            data.citations.forEach((citation, idx) => {
                const citationItem = document.createElement("div");
                citationItem.className = "citation-item";

                const citationHeader = document.createElement("div");
                citationHeader.className = "citation-header";

                const tagSpan = document.createElement("span");
                tagSpan.className = "citation-tag";
                const shortChunkId = citation.chunk_id ? citation.chunk_id.substring(0, 12) + "..." : `chunk_${idx+1}`;
                tagSpan.textContent = `[Source ${idx + 1}] ID: ${shortChunkId}`;

                const scoreSpan = document.createElement("span");
                scoreSpan.className = "citation-score";
                if (citation.score !== undefined) {
                    const scorePct = (citation.score * 100).toFixed(1);
                    scoreSpan.textContent = `Relevance: ${scorePct}%`;
                }

                citationHeader.appendChild(tagSpan);
                citationHeader.appendChild(scoreSpan);

                const citationSnippet = document.createElement("div");
                citationSnippet.className = "citation-snippet";
                citationSnippet.textContent = `"${citation.snippet}"`;

                citationItem.appendChild(citationHeader);
                citationItem.appendChild(citationSnippet);
                citationsList.appendChild(citationItem);
            });

            citationsContainer.appendChild(citationsList);
            messageCard.appendChild(citationsContainer);
        }

        // Render Latency & Token Usage Meta Row
        if (data.latency_ms !== undefined) {
            const metaDiv = document.createElement("div");
            metaDiv.className = "message-meta";

            const latencyPill = document.createElement("span");
            latencyPill.className = "meta-pill";
            latencyPill.textContent = `Latency: ${data.latency_ms.toFixed(1)} ms`;
            metaDiv.appendChild(latencyPill);

            if (data.retrieved_chunk_count !== undefined) {
                const chunkPill = document.createElement("span");
                chunkPill.className = "meta-pill";
                chunkPill.textContent = `Retrieved: ${data.retrieved_chunk_count} chunk(s)`;
                metaDiv.appendChild(chunkPill);
            }

            if (data.total_tokens) {
                const tokenPill = document.createElement("span");
                tokenPill.className = "meta-pill";
                tokenPill.textContent = `Tokens: ${data.total_tokens}`;
                metaDiv.appendChild(tokenPill);
            }

            messageCard.appendChild(metaDiv);
        }

        chatMessages.appendChild(messageCard);
        scrollToBottom();
    }

    // Helper: Scroll Chat Container to Bottom
    function scrollToBottom() {
        const chatViewport = document.querySelector(".chat-viewport");
        if (chatViewport) {
            chatViewport.scrollTop = chatViewport.scrollHeight;
        }
    }
});
