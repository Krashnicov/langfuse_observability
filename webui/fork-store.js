import { createStore, getStore } from "/js/AlpineStore.js";
import { callJsonApi } from "/js/api.js";

const API = "/plugins/langfuse-observability";

const model = {
  forking: false,
  error: "",
  _initialized: false,

  init() {
    if (this._initialized) return;
    this._initialized = true;
  },

  /**
   * Fork the currently selected chat context.
   * Returns the new context ID on success, or null on failure.
   */
  async forkCurrentChat() {
    const chatsStore = getStore("chats");
    if (!chatsStore || !chatsStore.selected) {
      this.error = "No chat selected";
      return null;
    }

    this.forking = true;
    this.error = "";

    try {
      const result = await callJsonApi(`${API}/chat_fork`, {
        context_id: chatsStore.selected,
      });

      if (!result.success) {
        this.error = result.error || "Fork failed";
        return null;
      }

      // Refresh the chat list so the fork appears
      if (chatsStore.loadContexts) {
        await chatsStore.loadContexts();
      }

      return result.new_context_id;
    } catch (e) {
      this.error = e.message || "Fork failed";
      return null;
    } finally {
      this.forking = false;
    }
  },

  /**
   * Fork the current chat and immediately open split view to compare.
   */
  async forkAndCompare() {
    const chatsStore = getStore("chats");
    if (!chatsStore || !chatsStore.selected) {
      this.error = "No chat selected";
      return;
    }

    const originalId = chatsStore.selected;
    const newId = await this.forkCurrentChat();
    if (!newId) return;

    // Open split view comparing original vs fork
    const splitView = getStore("splitView");
    if (splitView) {
      splitView.openSplit(originalId, newId);
    }
  },

  /**
   * Open split view comparing two existing contexts.
   */
  compareChats(leftId, rightId) {
    const splitView = getStore("splitView");
    if (splitView) {
      splitView.openSplit(leftId, rightId);
    }
  },
};

export const store = createStore("forkActions", model);
