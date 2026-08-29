import { useState, useRef, useEffect } from "react";

const DOCS = [
  { id: "d1", name: "AAPL_10K_2024.pdf", type: "10-K", size: "4.2 MB", date: "Jan 12" },
  { id: "d2", name: "NVDA_10Q_Q3.pdf", type: "10-Q", size: "2.8 MB", date: "Jan 9" },
  { id: "d3", name: "MSFT_Earnings_Q4.pdf", type: "Transcript", size: "1.1 MB", date: "Jan 5" },
  { id: "d4", name: "GOOGL_Proxy_2024.pdf", type: "Proxy", size: "3.4 MB", date: "Dec 28" },
  { id: "d5", name: "JPM_8K_Jan25.pdf", type: "8-K", size: "0.6 MB", date: "Dec 20" },
];

const INITIAL_CHATS = [
  {
    id: "c1",
    title: "Apple Revenue Analysis",
    preview: "Total net sales of $391B…",
    time: "2h ago",
    docs: ["d1"], // Included documents for this specific chat
    messages: [
      { id: "m1", role: "user", content: "What is Apple's revenue growth for FY2024?" },
      {
        id: "m2",
        role: "assistant",
        content:
          "Apple reported total net sales of $391.0 billion for FY2024, a 2% increase year-over-year. Services revenue was the standout, growing 13% to $96.2 billion — now 24.6% of total revenue. iPhone revenue dipped slightly by 0.4% to $201.2 billion.",
        citations: [
          { doc: "AAPL_10K_2024.pdf", page: 23, excerpt: "Net sales increased 2 percent or $7.7 billion during 2024 compared to 2023, driven primarily by higher net sales of Services and Mac." },
        ],
      },
    ],
  },
  {
    id: "c2",
    title: "NVIDIA Supply Chain",
    preview: "TSMC CoWoS bottleneck…",
    time: "Yesterday",
    docs: ["d2"],
    messages: [
      { id: "m3", role: "user", content: "How is NVIDIA managing its supply chain risks?" },
      {
        id: "m4",
        role: "assistant",
        content:
          "NVIDIA disclosed significant supply concentration in TSMC for wafer manufacturing. CoWoS advanced packaging is identified as a near-term bottleneck. Despite this, gross margins expanded to 74.6% in Q3, reflecting strong pricing power.",
        citations: [
          { doc: "NVDA_10Q_Q3.pdf", page: 14, excerpt: "We rely on TSMC to manufacture our products. The inability to procure sufficient CoWoS packaging has in the past limited our ability to meet customer demand." },
        ],
      },
    ],
  },
  {
    id: "c3",
    title: "MSFT Cloud Growth",
    preview: "Azure grew 33% YoY…",
    time: "2 days ago",
    docs: ["d3"],
    messages: [],
  },
];

const SUGGESTED = [
  "What were Apple's key risk factors?",
  "Compare NVDA and MSFT data center revenue",
  "Summarize JPMorgan's Q1 outlook",
  "What is Google's advertising trend?",
];

const typeColors = {
  "10-K": "bg-stone-100 text-stone-700 border-stone-300",
  "10-Q": "bg-amber-50 text-amber-700 border-amber-200",
  Transcript: "bg-teal-50 text-teal-700 border-teal-200",
  Proxy: "bg-rose-50 text-rose-700 border-rose-200",
  "8-K": "bg-stone-200 text-stone-800 border-stone-300",
};

function getGreeting() {
  const h = new Date().getHours();
  if (h < 5) return { label: "Night Owl!!"};
  if (h < 9) return { label: "Early Bird!!" };
  if (h < 12) return { label: "Good Morning", icon: "☀️"};
  if (h < 17) return { label: "Good Afternoon"};
  if (h < 21) return { label: "Good Evening"};
  return { label: "Night Owl"};
}

function ThemeToggle({ theme, onToggle }) {
  return (
    <button
      onClick={onToggle}
      className="w-9 h-9 rounded-full flex items-center justify-center transition-colors"
      style={{
        background: theme === "light" ? "#EBEAE4" : "#2A2925",
        color: theme === "light" ? "#1C1B19" : "#B1ADA1",
      }}
      title={theme === "light" ? "Switch to dark" : "Switch to light"}
    >
      {theme === "light" ? (
        <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
          <path d="M7.5 1v1.5M7.5 12.5V14M1 7.5h1.5M12.5 7.5H14M3.2 3.2l1.06 1.06M10.74 10.74l1.06 1.06M3.2 11.8l1.06-1.06M10.74 4.26l1.06-1.06M7.5 5a2.5 2.5 0 100 5 2.5 2.5 0 000-5z" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M12.5 9A6 6 0 015 1.5a6 6 0 100 11A6 6 0 0012.5 9z" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </button>
  );
}

function LandingPage({ onStart, theme, onToggleTheme }) {
  const dark = theme === "dark";
  const bg = dark ? "#1C1B19" : "#F4F3EE"; 
  const surface = dark ? "#2A2925" : "#FFFFFF";
  const border = dark ? "#38342E" : "#B1ADA1";
  const textMain = dark ? "#F4F3EE" : "#1C1B19";
  const textMuted = dark ? "#B1ADA1" : "#6B6660";
  const btnBg = dark ? "#F4F3EE" : "#1C1B19";
  const btnText = dark ? "#1C1B19" : "#F4F3EE";
  const accent = "#7C9A6E";

  return (
    <div className="min-h-full flex flex-col font-sans transition-colors duration-200" style={{ background: bg, color: textMain }}>
      <nav className="flex items-center justify-between px-10 py-5 border-b" style={{ borderColor: border }}>
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: textMain }}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <circle cx="7" cy="7" r="3" fill={bg} />
              <path d="M7 1v2M7 11v2M1 7h2M11 7h2" stroke={bg} strokeWidth="1.3" strokeLinecap="round" />
            </svg>
          </div>
          <span className="font-display font-bold text-lg tracking-tight" style={{ color: textMain }}>FinSage</span>
        </div>
        <div className="flex items-center gap-3">
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
          <button onClick={onStart} className="text-sm font-medium transition-colors" style={{ color: textMuted }}>
            Sign in
          </button>
        </div>
      </nav>

      <main className="flex-1 flex flex-col items-center justify-center px-6 text-center">
        <div className="inline-flex items-center gap-2 font-mono text-xs px-3 py-1.5 rounded-full border mb-8" style={{ background: surface, borderColor: border, color: textMuted }}>
          <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: accent }} />
          Beta · RAG-powered financial research
        </div>
        <h1 className="font-display text-5xl font-bold tracking-tight leading-tight max-w-2xl mb-6" style={{ color: textMain }}>
          Ask questions across<br />
          <span style={{ color: accent }}>every filing you own.</span>
        </h1>
        <p className="text-lg max-w-md mb-10 leading-relaxed" style={{ color: textMuted }}>
          Upload 10-Ks, earnings transcripts, and proxy statements. Get precise, cited answers from your entire document library in seconds.
        </p>
        <button onClick={onStart} className="font-semibold px-8 py-3.5 rounded-xl text-sm transition-all hover:opacity-90 shadow-sm" style={{ background: btnBg, color: btnText }}>
          Get Started
        </button>

        <div className="grid grid-cols-3 gap-4 mt-20 max-w-2xl w-full">
          {[
            { icon: "📄", title: "Multi-document RAG", desc: "Query across 10-Ks, 10-Qs, and transcripts simultaneously." },
            { icon: "🔗", title: "Cited answers", desc: "Every response references the exact page and excerpt." },
            { icon: "⚡", title: "Instant retrieval", desc: "Semantic search over millions of words in under 2 seconds." },
          ].map((f) => (
            <div key={f.title} className="rounded-xl p-5 text-left border" style={{ background: surface, borderColor: border }}>
              <div className="text-2xl mb-3">{f.icon}</div>
              <p className="font-semibold text-sm mb-1" style={{ color: textMain }}>{f.title}</p>
              <p className="text-xs leading-relaxed" style={{ color: textMuted }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}

function AuthPage({ onAuth, theme, onToggleTheme }) {
  const [tab, setTab] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  
  const dark = theme === "dark";
  const bg = dark ? "#1C1B19" : "#F4F3EE";
  const surface = dark ? "#2A2925" : "#FFFFFF";
  const border = dark ? "#38342E" : "#B1ADA1";
  const textMain = dark ? "#F4F3EE" : "#1C1B19";
  const textMuted = dark ? "#B1ADA1" : "#6B6660";
  const inputBg = dark ? "#1C1B19" : "#F4F3EE";
  const btnBg = dark ? "#F4F3EE" : "#1C1B19";
  const btnText = dark ? "#1C1B19" : "#F4F3EE";

  return (
    <div className="min-h-full flex flex-col items-center justify-center font-sans transition-colors duration-200" style={{ background: bg }}>
      <div className="absolute top-5 right-6">
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
      <div className="w-full max-w-md rounded-2xl shadow-sm p-8 border" style={{ background: surface, borderColor: border }}>
        <div className="flex items-center gap-2 mb-8">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: textMain }}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <circle cx="7" cy="7" r="3" fill={surface} />
              <path d="M7 1v2M7 11v2M1 7h2M11 7h2" stroke={surface} strokeWidth="1.3" strokeLinecap="round" />
            </svg>
          </div>
          <span className="font-display font-bold text-lg tracking-tight" style={{ color: textMain }}>FinSage</span>
        </div>

        <div className="flex rounded-lg p-1 mb-6" style={{ background: bg }}>
          {["login", "signup"].map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="flex-1 py-2 text-sm font-medium rounded-md transition-all capitalize"
              style={tab === t ? { background: surface, color: textMain, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" } : { color: textMuted }}
            >
              {t === "login" ? "Sign in" : "Sign up"}
            </button>
          ))}
        </div>

        <div className="space-y-4">
          {tab === "signup" && (
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: textMuted }}>Full name</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Alex Chen"
                className="w-full px-3 py-2.5 rounded-lg border text-sm outline-none transition-all"
                style={{ background: inputBg, borderColor: border, color: textMain }}
              />
            </div>
          )}
          <div>
            <label className="block text-xs font-medium mb-1.5" style={{ color: textMuted }}>Email</label>
            <input
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="alex@fund.com"
              type="email"
              className="w-full px-3 py-2.5 rounded-lg border text-sm outline-none transition-all"
              style={{ background: inputBg, borderColor: border, color: textMain }}
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1.5" style={{ color: textMuted }}>Password</label>
            <input
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              type="password"
              className="w-full px-3 py-2.5 rounded-lg border text-sm outline-none transition-all"
              style={{ background: inputBg, borderColor: border, color: textMain }}
            />
          </div>
          <button
            onClick={onAuth}
            className="w-full font-semibold py-2.5 rounded-lg text-sm transition-all hover:opacity-90 mt-2"
            style={{ background: btnBg, color: btnText }}
          >
            {tab === "login" ? "Sign in" : "Create account"}
          </button>
        </div>
        {tab === "login" && (
          <p className="text-xs text-center mt-4" style={{ color: textMuted }}>
            Don't have an account? <button onClick={() => setTab("signup")} className="underline" style={{ color: textMain }}>Sign up</button>
          </p>
        )}
      </div>
    </div>
  );
}

function CitationCard({ citation, dark }) {
  const docObj = DOCS.find((d) => d.name === citation.doc);
  const docType = docObj ? docObj.type : "DOC";
  const surface = dark ? "#2A2925" : "#FFFFFF";
  const border = dark ? "#38342E" : "#B1ADA1";
  const textMuted = dark ? "#B1ADA1" : "#6B6660";

  return (
    <div className="rounded-lg p-3 border text-xs mt-2" style={{ background: surface, borderColor: border }}>
      <div className="flex items-center gap-2 mb-2">
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${typeColors[docType] || ""}`}>{docType}</span>
        <span className="font-mono" style={{ color: textMuted }}>{citation.doc} · p.{citation.page}</span>
      </div>
      <p className="leading-relaxed border-l-2 pl-2 italic" style={{ borderColor: "#7C9A6E", color: textMuted }}>
        "{citation.excerpt}"
      </p>
    </div>
  );
}

function MessageBubble({ msg, dark }) {
  const [expanded, setExpanded] = useState(false);
  const textMain = dark ? "#F4F3EE" : "#1C1B19";
  const textMuted = dark ? "#B1ADA1" : "#6B6660";
  const surface = dark ? "#2A2925" : "#FFFFFF";
  const border = dark ? "#38342E" : "#B1ADA1";
  const accent = "#7C9A6E";

  if (msg.role === "user") {
    return (
      <div className="flex justify-end mb-4">
        <div
          className="max-w-[65%] rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm leading-relaxed"
          style={{ background: dark ? "#38342E" : "#1C1B19", color: dark ? "#F4F3EE" : "#FFFFFF" }}
        >
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 mb-5">
      <div
        className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5 text-sm"
        style={{ background: surface, border: `1px solid ${border}`, color: textMuted }}
      >
        ✦
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm leading-relaxed mb-2" style={{ color: textMain }}>{msg.content}</p>
        {msg.citations && msg.citations.length > 0 && (
          <>
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-1 text-xs font-medium transition-colors"
              style={{ color: accent }}
            >
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ transform: expanded ? "rotate(90deg)" : undefined, transition: "transform 0.15s" }}>
                <path d="M3 2l4 3-4 3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              {msg.citations.length} source{msg.citations.length > 1 ? "s" : ""}
            </button>
            {expanded && msg.citations.map((c, i) => <CitationCard key={i} citation={c} dark={dark} />)}
          </>
        )}
      </div>
    </div>
  );
}

function MainApp({ onLogout, theme, onToggleTheme }) {
  const dark = theme === "dark";
  const [chats, setChats] = useState(INITIAL_CHATS);
  const [activeChatId, setActiveChatId] = useState("c1");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [showRightSidebar, setShowRightSidebar] = useState(true);
  const [sampleIdx, setSampleIdx] = useState(0);
  
  const bottomRef = useRef(null);
  const profileRef = useRef(null);
  const fileInputRef = useRef(null);
  const greeting = getGreeting();

  const activeChat = chats.find((c) => c.id === activeChatId);
  const currentDocs = activeChat.docs ? DOCS.filter(d => activeChat.docs.includes(d.id)) : [];

  const bg = dark ? "#1C1B19" : "#F4F3EE";
  const surface = dark ? "#2A2925" : "#FFFFFF";
  const border = dark ? "#38342E" : "#B1ADA1";
  const textMain = dark ? "#F4F3EE" : "#1C1B19";
  const textMuted = dark ? "#B1ADA1" : "#6B6660";
  const sidebarBg = dark ? "#1E1D1A" : "#F4F3EE";
  const btnBg = dark ? "#F4F3EE" : "#1C1B19";
  const btnText = dark ? "#1C1B19" : "#F4F3EE";
  const accent = "#7C9A6E";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChatId, loading]);

  useEffect(() => {
    function handler(e) {
      if (profileRef.current && !profileRef.current.contains(e.target)) setProfileOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const SAMPLE_RESPONSES = [
    {
      content: "Based on the filings, Apple's Services segment grew 13% YoY to $96.2B, now representing nearly 25% of total revenue. This shift toward recurring, high-margin revenue is a key strategic inflection.",
      citations: [{ doc: "AAPL_10K_2024.pdf", page: 31, excerpt: "Services net sales were $96,169 million and $85,200 million for 2024 and 2023, respectively." }],
    },
    {
      content: "NVIDIA's data center revenue reached $47.5B in Q3 FY2025, up 112% year-over-year. The Hopper GPU architecture continues to dominate AI training workloads.",
      citations: [{ doc: "NVDA_10Q_Q3.pdf", page: 8, excerpt: "Data Center revenue was $30.8 billion for Q3, compared with $14.5 billion in the prior year quarter." }],
    },
  ];

  function send(text) {
    if (!text.trim() || loading) return;
    const userMsg = { id: Date.now().toString(), role: "user", content: text.trim() };
    setChats((prev) =>
      prev.map((c) => c.id === activeChatId ? { ...c, messages: [...c.messages, userMsg], preview: text.trim().slice(0, 40) + "…" } : c)
    );
    setInput("");
    setLoading(true);
    const sample = SAMPLE_RESPONSES[sampleIdx % SAMPLE_RESPONSES.length];
    setSampleIdx((i) => i + 1);
    
    setTimeout(() => {
      const assistantMsg = { id: (Date.now() + 1).toString(), role: "assistant", ...sample };
      setChats((prev) => prev.map((c) => c.id === activeChatId ? { ...c, messages: [...c.messages, assistantMsg] } : c));
      setLoading(false);
    }, 1300);
  }

  function newChat() {
    const id = "c" + Date.now();
    setChats((prev) => [{ id, title: "New chat", preview: "Ask anything…", time: "now", docs: [], messages: [] }, ...prev]);
    setActiveChatId(id);
  }

  return (
    <div className="flex h-full font-sans transition-colors duration-200" style={{ background: bg }}>
      <aside className="w-60 flex flex-col border-r shrink-0" style={{ background: sidebarBg, borderColor: border }}>
        <div className="flex items-center justify-between px-4 py-4 border-b" style={{ borderColor: border }}>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: textMain }}>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <circle cx="6" cy="6" r="2.5" fill={sidebarBg} />
                <path d="M6 1v1.5M6 9.5V11M1 6h1.5M9.5 6H11" stroke={sidebarBg} strokeWidth="1.2" strokeLinecap="round" />
              </svg>
            </div>
            <span className="font-display font-bold text-sm" style={{ color: textMain }}>FinSage</span>
          </div>
          <button
            onClick={newChat}
            className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
            style={{ background: surface, color: textMuted, border: `1px solid ${border}` }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M6 1v10M1 6h10" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto py-2">
          <p className="font-mono text-[10px] uppercase tracking-widest px-4 py-2" style={{ color: textMuted }}>Chats</p>
          {chats.map((c) => {
            const active = activeChatId === c.id;
            return (
              <button
                key={c.id}
                onClick={() => setActiveChatId(c.id)}
                className="w-full text-left px-3 py-2.5 transition-colors mx-1 rounded-lg"
                style={{
                  background: active ? surface : "transparent",
                  color: active ? textMain : textMuted,
                  width: "calc(100% - 8px)",
                }}
              >
                <div className="flex items-center justify-between mb-0.5">
                  <span className="text-xs font-semibold truncate max-w-[120px]" style={{ color: active ? textMain : textMuted }}>{c.title}</span>
                  <span className="text-[10px] font-mono" style={{ color: textMuted }}>{c.time}</span>
                </div>
                <p className="text-[11px] truncate" style={{ color: textMuted }}>{c.preview}</p>
              </button>
            );
          })}
        </div>

        <div className="px-3 py-3 border-t relative" style={{ borderColor: border }} ref={profileRef}>
          <button
            onClick={() => setProfileOpen(!profileOpen)}
            className="flex items-center gap-2 w-full rounded-lg p-2 transition-colors hover:opacity-80"
            >
            <div className="w-9 h-9 rounded-full flex items-center justify-center text-white text-sm font-bold shrink-0" style={{ background: "#888073" }}>A</div>
            <div className="flex-1 min-w-0 text-left pl-1">
                <p className="text-sm font-medium truncate leading-tight" style={{ color: textMain }}>Alex Chen</p>
                <p className="text-xs truncate mt-0.5" style={{ color: textMuted }}>alex@fund.com</p>
            </div>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: textMuted }} className="shrink-0">
                <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </button>
          
          {profileOpen && (
            <div className="absolute bottom-full left-2 right-2 mb-2 rounded-xl shadow-lg border py-1 z-50" style={{ background: surface, borderColor: border }}>
              <button onClick={onToggleTheme} className="w-full text-left px-3 py-2 text-xs" style={{ color: textMuted }}>
                {dark ? "Light mode" : "Dark mode"}
              </button>
              <button onClick={onLogout} className="w-full text-left px-3 py-2 text-xs text-red-500">
                Log out
              </button>
            </div>
          )}
        </div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center justify-between px-6 py-3 shrink-0" style={{ background: bg }}>
            <div>
                <p className="font-semibold text-sm" style={{ color: textMain }}>{activeChat.title}</p>
                <p className="text-xs" style={{ color: textMuted }}>{currentDocs.length > 0 ? `${currentDocs.length} docs included` : "All documents"}</p>
            </div>
            <div className="flex items-center gap-2">
                <ThemeToggle theme={theme} onToggle={onToggleTheme} />
                <div className="font-mono text-[11px] px-2 py-1 rounded-md" style={{ background: surface, color: textMuted }}>
                GPT-4o · RAG
                </div>
                {!showRightSidebar && (
                <button 
                    onClick={() => setShowRightSidebar(true)}
                    className="w-9 h-9 flex items-center justify-center rounded-full transition-colors hover:opacity-80 ml-2"
                    style={{ color: textMuted, background: surface, border: `1px solid ${border}` }}
                    title="Show Sidebar"
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="15" y1="3" x2="15" y2="21"></line>
                    </svg>
                </button>
                )}
            </div>
        </header>

        <div className="flex-1 overflow-y-auto px-6 py-5" style={{ background: bg }}>
          {activeChat.messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <p className="text-3xl mb-1">{greeting.icon}</p>
              <p className="font-display text-xl font-semibold mb-1" style={{ color: textMain }}>{greeting.label},</p>
              <p className="text-sm mb-8" style={{ color: textMuted }}>Ask the filings. Every answer comes with a source.</p>
              <div className="grid grid-cols-2 gap-2 max-w-md w-full">
                {SUGGESTED.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="text-left text-xs px-3 py-2.5 rounded-lg border transition-colors leading-relaxed"
                    style={{ background: surface, borderColor: border, color: textMuted }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-2xl mx-auto">
              {activeChat.messages.map((m) => <MessageBubble key={m.id} msg={m} dark={dark} />)}
              {loading && (
                <div className="flex gap-3 mb-4">
                  <div className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 text-sm" style={{ background: surface, border: `1px solid ${border}`, color: textMuted }}>✦</div>
                  <div className="flex items-center gap-1.5 text-xs" style={{ color: textMuted }}>
                    <span>Retrieving sources</span>
                    <span className="w-1 h-1 rounded-full animate-pulse" style={{ background: accent }} />
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="px-6 py-4 shrink-0" style={{ background: bg }}>
          <div className="max-w-2xl mx-auto">
            <div className="flex flex-col border rounded-xl transition-colors" style={{ borderColor: border, background: surface }}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }}
                placeholder="Ask about revenue, risk factors, guidance…"
                rows={1}
                className="flex-1 bg-transparent px-4 pt-3 pb-1 text-sm outline-none resize-none leading-relaxed w-full"
                style={{ color: textMain, maxHeight: 120 }}
              />
              <div className="flex items-center justify-between px-3 pb-2.5">
                <div className="flex items-center gap-1">
                  <input ref={fileInputRef} type="file" accept=".pdf,.doc,.docx,.txt" className="hidden" multiple />
                  <button onClick={() => fileInputRef.current?.click()} className="flex items-center justify-center w-8 h-8 rounded-lg transition-colors hover:opacity-80" style={{ color: textMuted, background: bg }} title="Upload File">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="5" x2="12" y2="19"></line>
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                    </svg>
                  </button>
                </div>
                <button
                  onClick={() => send(input)}
                  disabled={!input.trim() || loading}
                  className="w-8 h-8 rounded-lg flex items-center justify-center transition-all hover:opacity-90 disabled:opacity-25"
                  style={{ background: btnBg }}
                >
                  <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                    <path d="M1.5 6.5h10M6.5 1.5l5 5-5 5" stroke={btnText} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {showRightSidebar && (
        <aside className="w-56 flex flex-col border-l shrink-0" style={{ background: sidebarBg, borderColor: border }}>
          <div className="px-4 py-4 border-b" style={{ borderColor: border }}>
            <div className="flex items-center justify-between">
                <p className="font-semibold text-xs" style={{ color: textMain }}>Document Library</p>
                <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] px-1.5 py-0.5 rounded" style={{ background: surface, color: textMuted }}>{currentDocs.length}</span>
                <button 
                    onClick={() => setShowRightSidebar(false)}
                    className="flex items-center justify-center transition-colors hover:opacity-80"
                    style={{ color: textMuted }}
                    title="Hide Sidebar"
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="15" y1="3" x2="15" y2="21"></line>
                    </svg>
                </button>
                </div>
             </div>
          </div>
          <div className="flex-1 overflow-y-auto py-2">
            {currentDocs.length === 0 ? (
              <p className="text-xs text-center mt-4" style={{ color: textMuted }}>No documents included.</p>
            ) : (
              currentDocs.map((doc) => (
                <div key={doc.id} className="w-full text-left px-3 py-2.5 mb-0.5">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span className={`text-[9px] font-mono px-1 py-0.5 rounded border ${typeColors[doc.type] || ""}`}>{doc.type}</span>
                  </div>
                  <p className="text-[11px] font-medium leading-tight truncate" style={{ color: textMain }}>{doc.name}</p>
                  <p className="text-[10px] mt-0.5 font-mono" style={{ color: textMuted }}>{doc.size} · {doc.date}</p>
                </div>
              ))
            )}
          </div>
        </aside>
      )}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("landing");
  const [theme, setTheme] = useState("light");
  
  const toggle = () => setTheme((t) => (t === "light" ? "dark" : "light"));

  return (
    <div className="h-full">
      {page === "landing" && <LandingPage onStart={() => setPage("auth")} theme={theme} onToggleTheme={toggle} />}
      {page === "auth" && <AuthPage onAuth={() => setPage("app")} theme={theme} onToggleTheme={toggle} />}
      {page === "app" && <MainApp onLogout={() => setPage("landing")} theme={theme} onToggleTheme={toggle} />}
    </div>
  );
}