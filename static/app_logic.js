/**
 * MedPulse AI - Frontend Logic
 * Implements Chat History, Search, Markdown, and Premium SaaS Features
 */

class MedPulseApp {
    constructor() {
        this.chats = JSON.parse(localStorage.getItem('medpulse_chats')) || [];
        this.currentChatId = null;
        this.isTyping = false;
        
        this.init();
    }

    init() {
        // DOM Elements
        this.input = document.getElementById('user-input');
        this.sendBtn = document.getElementById('send-btn');
        this.messageContainer = document.getElementById('message-container');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.emptyState = document.getElementById('empty-state');
        this.chatViewport = document.getElementById('chat-viewport');
        this.sidebar = document.getElementById('main-sidebar');
        this.overlay = document.getElementById('sidebar-overlay');
        
        // Event Listeners
        this.input.addEventListener('input', () => this.handleInput());
        this.input.addEventListener('keydown', (e) => this.handleKeydown(e));
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        document.getElementById('open-sidebar').addEventListener('click', () => this.toggleSidebar(true));
        document.getElementById('close-sidebar').addEventListener('click', () => this.toggleSidebar(false));
        this.overlay.addEventListener('click', () => this.toggleSidebar(false));

        // Configure Markdown
        marked.setOptions({
            highlight: function(code, lang) {
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language }).value;
            },
            langPrefix: 'hljs language-',
            breaks: true
        });

        // Load Initial Theme
        if (localStorage.getItem('theme') === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
        }

        // Render Sidebar
        this.renderHistory();
        
        // Final UI adjustments
        this.input.focus();
    }

    // --- State Management ---

    createNewChat() {
        this.currentChatId = Date.now().toString();
        this.clearChatDisplay();
        this.updateHeader('New Consultation');
        this.toggleSidebar(false);
        this.input.focus();
    }

    clearChatDisplay() {
        this.messageContainer.innerHTML = '';
        this.emptyState.classList.remove('hidden');
    }

    updateHeader(title) {
        document.getElementById('current-chat-title').innerText = title;
    }

    // --- Communication ---

    async sendMessage() {
        const query = this.input.value.trim();
        if (!query || this.isTyping) return;

        // 1. Prepare UI
        if (this.emptyState) this.emptyState.classList.add('hidden');
        if (!this.currentChatId) this.currentChatId = Date.now().toString();

        this.addMessage('user', query);
        this.input.value = '';
        this.handleInput(); // Reset height
        this.setLoading(true);

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();
            this.setLoading(false);
            
            if (data.answer) {
                this.addMessage('bot', data.answer);
                this.saveChat(query, data.answer);
            } else {
                this.addMessage('bot', "I encountered an error processing your request. Please try again.");
            }
        } catch (error) {
            this.setLoading(false);
            this.addMessage('bot', "Server unreachable. Ensure the backend is running.");
        }
    }

    addMessage(role, text) {
        const row = document.createElement('div');
        row.className = `message-row role-${role}`;
        
        const avatarIcon = role === 'bot' ? 'stethoscope' : 'user';
        const avatarClass = role === 'bot' ? 'bot-avatar' : 'user-avatar';
        
        // Detect medical risks (example logic)
        let processedText = text;
        const risks = ["malignant", "cancer", "critical", "urgent", "spiculated", "metastasis"];
        risks.forEach(risk => {
            const regex = new RegExp(`\\b${risk}\\b`, 'gi');
            processedText = processedText.replace(regex, `<span class="risk-highlight">${risk}</span>`);
        });

        row.innerHTML = `
            <div class="msg-avatar ${avatarClass}">
                <i class="fas fa-${avatarIcon}"></i>
            </div>
            <div class="msg-content">
                <div class="msg-text">${marked.parse(processedText)}</div>
                ${role === 'bot' ? this.getMessageActions() : ''}
            </div>
        `;

        this.messageContainer.appendChild(row);
        this.scrollToBottom();
        
        // Highlight code blocks
        row.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    }

    getMessageActions() {
        return `
            <div class="msg-actions">
                <button class="action-btn" onclick="chatApp.copyToClipboard(this)">
                    <i class="fas fa-copy"></i> Copy
                </button>
                <button class="action-btn">
                    <i class="fas fa-rotate"></i> Regenerate
                </button>
                <button class="action-btn">
                    <i class="fas fa-bookmark"></i> Save
                </button>
            </div>
        `;
    }

    // --- History & Storage ---

    saveChat(userQuery, botResponse) {
        let chat = this.chats.find(c => c.id === this.currentChatId);
        
        if (!chat) {
            chat = {
                id: this.currentChatId,
                title: userQuery.length > 30 ? userQuery.substring(0, 30) + '...' : userQuery,
                messages: [],
                timestamp: Date.now()
            };
            this.chats.unshift(chat);
        }

        chat.messages.push({ role: 'user', content: userQuery });
        chat.messages.push({ role: 'bot', content: botResponse });
        chat.timestamp = Date.now(); // Update for sorting

        localStorage.setItem('medpulse_chats', JSON.stringify(this.chats));
        this.renderHistory();
        this.updateHeader(chat.title);
    }

    renderHistory(filter = '') {
        const todayList = document.getElementById('list-today');
        const yesterdayList = document.getElementById('list-yesterday');
        const olderList = document.getElementById('list-older');

        todayList.innerHTML = '';
        yesterdayList.innerHTML = '';
        olderList.innerHTML = '';

        const now = new Date();
        const oneDay = 24 * 60 * 60 * 1000;

        const filteredChats = this.chats.filter(c => 
            c.title.toLowerCase().includes(filter.toLowerCase()) || 
            c.messages.some(m => m.content.toLowerCase().includes(filter.toLowerCase()))
        );

        filteredChats.forEach(chat => {
            const date = new Date(chat.timestamp);
            const diff = now - date;

            const item = document.createElement('div');
            item.className = `history-item ${this.currentChatId === chat.id ? 'active' : ''}`;
            item.onclick = () => this.loadChat(chat.id);
            
            const lastMsg = chat.messages[chat.messages.length - 1]?.content || '';

            item.innerHTML = `
                <i class="fas fa-message"></i>
                <div class="chat-info">
                    <span class="chat-name">${chat.title}</span>
                    <span class="chat-preview">${lastMsg.substring(0, 35)}...</span>
                </div>
                <div class="item-actions">
                    <i class="fas fa-ellipsis-h" onclick="event.stopPropagation(); chatApp.showItemMenu('${chat.id}')"></i>
                </div>
            `;

            if (diff < oneDay) todayList.appendChild(item);
            else if (diff < 2 * oneDay) yesterdayList.appendChild(item);
            else olderList.appendChild(item);
        });

        // Hide empty sections
        [todayList, yesterdayList, olderList].forEach(list => {
            const section = list.closest('.history-section');
            if (list.children.length === 0) section.classList.add('hidden');
            else section.classList.remove('hidden');
        });
    }

    loadChat(id) {
        const chat = this.chats.find(c => c.id === id);
        if (!chat) return;

        this.currentChatId = id;
        this.clearChatDisplay();
        this.emptyState.classList.add('hidden');
        this.updateHeader(chat.title);

        chat.messages.forEach(msg => {
            this.addMessage(msg.role, msg.content);
        });

        this.renderHistory();
        this.toggleSidebar(false);
    }

    filterChats(query) {
        this.renderHistory(query);
    }

    // --- UI Helpers ---

    handleInput() {
        this.input.style.height = 'auto';
        this.input.style.height = (this.input.scrollHeight) + 'px';
        this.sendBtn.disabled = this.input.value.trim().length === 0;
    }

    handleKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    setLoading(loading) {
        this.isTyping = loading;
        if (loading) {
            this.typingIndicator.classList.remove('hidden');
            this.scrollToBottom();
        } else {
            this.typingIndicator.classList.add('hidden');
        }
    }

    scrollToBottom() {
        this.chatViewport.scrollTo({
            top: this.chatViewport.scrollHeight,
            behavior: 'smooth'
        });
    }

    toggleSidebar(open) {
        if (open) {
            this.sidebar.classList.add('open');
            this.overlay.classList.add('active');
        } else {
            this.sidebar.classList.remove('open');
            this.overlay.classList.remove('active');
        }
    }

    toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const target = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', target);
        localStorage.setItem('theme', target);
        
        const icon = document.querySelector('.theme-toggle i');
        icon.className = target === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    sendQuickPrompt(text) {
        this.input.value = text;
        this.handleInput();
        this.sendMessage();
    }

    copyToClipboard(btn) {
        const text = btn.closest('.msg-content').querySelector('.msg-text').innerText;
        navigator.clipboard.writeText(text).then(() => {
            const original = btn.innerHTML;
            btn.innerHTML = `<i class="fas fa-check"></i> Copied!`;
            setTimeout(() => btn.innerHTML = original, 2000);
        });
    }

    exportAsPDF() { alert("Clinical PDF Export generated (Simulation)"); }
    shareChat() { alert("Encrypted consultation link copied to clipboard (Simulation)"); }
}

// Global instance
const chatApp = new MedPulseApp();
