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

    let isSubmitting = false;

    // API Endpoint Configuration
    const API_ENDPOINT = "/api/v1/query";

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
                    // Ignore JSON parsing failure for non-JSON response
                }
                throw new Error(errorDetail);
            }

            const data = await response.json();
            appendAssistantMessage(data);

        } catch (error) {
            showError(`Failed to process request: ${error.message}`);
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
            loadingIndicator.style.display = "block";
            statusIndicator.textContent = "Generating...";
            statusIndicator.style.backgroundColor = "#f39c12";
        } else {
            loadingIndicator.style.display = "none";
            statusIndicator.textContent = "Ready";
            statusIndicator.style.backgroundColor = "#27ae60";
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
        contentDiv.textContent = data.answer || "No response generated.";

        messageCard.appendChild(senderLabel);
        messageCard.appendChild(contentDiv);

        // Render Citations if present
        if (data.citations && data.citations.length > 0) {
            const citationsContainer = document.createElement("div");
            citationsContainer.className = "citations-container";

            const citationsTitle = document.createElement("div");
            citationsTitle.className = "citations-title";
            citationsTitle.textContent = `Retrieved Sources (${data.citations.length}):`;
            citationsContainer.appendChild(citationsTitle);

            data.citations.forEach((citation, idx) => {
                const citationItem = document.createElement("div");
                citationItem.className = "citation-item";

                const citationHeader = document.createElement("div");
                citationHeader.className = "citation-header";
                const scoreText = citation.score ? ` (Score: ${citation.score.toFixed(4)})` : "";
                citationHeader.textContent = `[${idx + 1}] Chunk ID: ${citation.chunk_id}${scoreText}`;

                const citationSnippet = document.createElement("div");
                citationSnippet.className = "citation-snippet";
                citationSnippet.textContent = `"${citation.snippet}"`;

                citationItem.appendChild(citationHeader);
                citationItem.appendChild(citationSnippet);
                citationsContainer.appendChild(citationItem);
            });

            messageCard.appendChild(citationsContainer);
        }

        // Render Latency & Token Usage Meta if available
        if (data.latency_ms !== undefined) {
            const metaDiv = document.createElement("div");
            metaDiv.className = "message-meta";
            let metaText = `Latency: ${data.latency_ms.toFixed(1)} ms`;
            if (data.total_tokens) {
                metaText += ` | Tokens: ${data.total_tokens}`;
            }
            metaDiv.textContent = metaText;
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
