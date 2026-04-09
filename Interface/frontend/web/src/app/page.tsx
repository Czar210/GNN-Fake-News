"use client";
import { useState, useEffect } from "react";
import { Search, ShieldAlert, ShieldCheck, Activity, BarChart3, Clock, Globe } from "lucide-react";

// ─── Visualização do Grafo de Propagação ──────────────────────────────────────
function GraphViz({ explanation, isFake }: { explanation: any; isFake: boolean }) {
  if (!explanation || !explanation.edges || explanation.edges.length === 0) {
    return (
      <p className="text-center text-gray-500 text-sm py-6">
        Nó único — sem interações para visualizar.
      </p>
    );
  }

  const W = 520, H = 300;
  const cx = W / 2, cy = H / 2;
  const RADIUS = 115;

  // Limita a 12 filhos para não poluir o SVG
  const edges: Array<{ from: number; to: number; importance: number }> =
    explanation.edges.slice(0, 12);
  const numChildren = edges.length;

  const childPos = edges.map((_, i) => {
    const angle = (2 * Math.PI * i) / numChildren - Math.PI / 2;
    return { x: cx + RADIUS * Math.cos(angle), y: cy + RADIUS * Math.sin(angle) };
  });

  const edgeColor = (imp: number) =>
    imp > 0.66 ? (isFake ? "#ef4444" : "#10b981")
    : imp > 0.33 ? "#f59e0b"
    : "#374151";

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full max-w-lg mx-auto"
      aria-label="Grafo de propagação"
    >
      {/* Arestas */}
      {edges.map((e, i) => {
        const imp = e.importance;
        const { x, y } = childPos[i];
        return (
          <line
            key={i}
            x1={cx} y1={cy}
            x2={x}  y2={y}
            stroke={edgeColor(imp)}
            strokeWidth={1 + imp * 5}
            strokeOpacity={0.35 + imp * 0.65}
          />
        );
      })}

      {/* Nós filhos (interações) */}
      {edges.map((e, i) => {
        const imp = e.importance;
        const { x, y } = childPos[i];
        const highlight = imp > 0.5;
        return (
          <g key={i}>
            <circle
              cx={x} cy={y} r={15}
              fill={highlight ? (isFake ? "#7f1d1d" : "#064e3b") : "#1f2937"}
              stroke={edgeColor(imp)}
              strokeWidth={2}
            />
            <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle"
              fill="white" fontSize={8} fontFamily="monospace">
              {e.to}
            </text>
            <title>{`Nó ${e.to} — influência: ${(imp * 100).toFixed(0)}%`}</title>
          </g>
        );
      })}

      {/* Nó raiz */}
      <circle cx={cx} cy={cy} r={24} fill="#1e3a5f" stroke="#3b82f6" strokeWidth={2.5} />
      <text x={cx} y={cy - 4} textAnchor="middle" fill="white" fontSize={9} fontWeight="bold">POST</text>
      <text x={cx} y={cy + 8} textAnchor="middle" fill="#93c5fd" fontSize={7}>raiz</text>
    </svg>
  );
}

// ─── Destaque de palavras por importância BERT ────────────────────────────────
function WordHighlight({ words }: { words: Array<{ word: string; importance: number }> }) {
  if (!words || words.length === 0) return null;

  const max = Math.max(...words.map((w) => w.importance), 0.001);
  // Reconstrói a ordem original (o array vem sorted, reordenamos por posição)
  // A API retorna na ordem original (sorted só para exibição na legenda)
  // Para o highlight precisamos da ordem do texto → usamos o índice original
  // Como a API os ordena por importância, vamos só exibir o mapa de cores
  const byWord = new Map(words.map((w) => [w.word, w.importance]));
  // Pega todas as palavras na ordem em que aparecem (reconstruída do array)
  const inOrder = [...words].sort((a, b) => {
    // fallback: exibe na ordem em que vieram (já é a ordem do texto na API)
    return 0;
  });

  return (
    <p className="text-sm leading-loose text-gray-200 font-mono flex flex-wrap gap-1">
      {inOrder.map((w, i) => {
        const intensity = w.importance / max;
        const bg =
          intensity > 0.6
            ? `rgba(239,68,68,${(intensity * 0.45).toFixed(2)})`
            : intensity > 0.25
            ? `rgba(245,158,11,${(intensity * 0.55).toFixed(2)})`
            : "transparent";
        return (
          <span
            key={i}
            style={{ backgroundColor: bg, borderRadius: "4px", padding: "1px 4px" }}
            title={`Δ cosseno: ${(w.importance * 100).toFixed(2)}%`}
          >
            {w.word}
          </span>
        );
      })}
    </p>
  );
}

// ─── Legenda compacta ──────────────────────────────────────────────────────────
function GraphLegend({ isFake }: { isFake: boolean }) {
  const color = isFake ? "bg-red-500" : "bg-emerald-500";
  return (
    <div className="flex items-center justify-center gap-6 text-xs text-gray-400 mt-3">
      <span className="flex items-center gap-1.5">
        <span className={`inline-block w-3 h-1.5 rounded-full ${color}`} /> Alta influência
      </span>
      <span className="flex items-center gap-1.5">
        <span className="inline-block w-3 h-1.5 rounded-full bg-amber-500" /> Média
      </span>
      <span className="flex items-center gap-1.5">
        <span className="inline-block w-3 h-1.5 rounded-full bg-gray-600" /> Baixa
      </span>
    </div>
  );
}

const API_URL = "http://localhost:8000";

export default function Home() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    fetchHistory();
  }, []);

  async function fetchHistory() {
    try {
      const res = await fetch(`${API_URL}/api/history`);
      if (res.ok) setHistory(await res.json());
    } catch (e) { console.error(e); }
  }

  async function analyze() {
    if (!url) return;
    setLoading(true);
    setResult(null);
    try {
      // 1. Dispara a análise e recebe o task_id imediatamente
      const res = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });
      const { task_id } = await res.json();

      // 2. Polling até o resultado ficar pronto
      await new Promise<void>((resolve) => {
        const interval = setInterval(async () => {
          try {
            const poll = await fetch(`${API_URL}/api/result/${task_id}`);
            const data = await poll.json();
            if (data.status === "done") {
              clearInterval(interval);
              setResult(data);
              fetchHistory();
              resolve();
            } else if (data.status === "error") {
              clearInterval(interval);
              console.error("Erro na análise:", data);
              resolve();
            }
          } catch (e) {
            clearInterval(interval);
            console.error(e);
            resolve();
          }
        }, 1500);
      });
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }

  return (
    <main className="min-h-screen bg-[#000000] text-gray-100 font-sans selection:bg-blue-500/30">
      {/* Background gradients iPhone style */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-900/30 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-indigo-900/20 blur-[100px]" />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto p-6 pt-16">
        <header className="mb-14 text-center space-y-4">
          <div className="inline-flex items-center justify-center p-3 bg-white/5 rounded-2xl border border-white/10 mb-4 backdrop-blur-xl">
             <Globe className="w-8 h-8 text-blue-400" />
          </div>
          <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight bg-gradient-to-br from-white via-gray-200 to-gray-500 bg-clip-text text-transparent">
            Neural Truth Engine
          </h1>
          <p className="text-gray-400 text-lg max-w-xl mx-auto font-medium">
            Detecção avançada de Fake News no Bluesky orientada por Grafos e Votação Multimodelo (Ensemble).
          </p>
        </header>

        {/* Search Bar (Dynamic Island Style) */}
        <div className="max-w-3xl mx-auto bg-white/5 border border-white/10 p-2 rounded-[2rem] shadow-2xl backdrop-blur-xl flex items-center transition-all focus-within:ring-4 focus-within:ring-blue-500/20 focus-within:border-blue-500/40">
          <div className="pl-6 text-gray-400">
            <Search className="w-6 h-6" />
          </div>
          <input 
            type="text" 
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && analyze()}
            placeholder="Cole o link do post do Bluesky..."
            className="flex-1 bg-transparent border-none outline-none text-white px-6 py-4 text-lg placeholder-gray-500"
          />
          <button 
            onClick={analyze}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-500 text-white px-8 py-4 rounded-full font-bold transition-all disabled:opacity-50 disabled:scale-95 active:scale-95 flex items-center gap-2"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : "Analisar"}
          </button>
        </div>

        {/* Main Result Area */}
        {result && (
          <div className="mt-12 animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className={`p-1 rounded-3xl bg-gradient-to-b ${result.is_fake ? 'from-red-500/30 to-rose-900/10' : 'from-emerald-500/30 to-teal-900/10'}`}>
              <div className="bg-[#0a0f1a]/80 backdrop-blur-2xl p-8 rounded-[22px] border border-white/5 shadow-2xl relative overflow-hidden">
                
                <div className="flex flex-col md:flex-row items-center gap-8 z-10 relative">
                  <div className={`p-6 rounded-full flex-shrink-0 ${result.is_fake ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                    {result.is_fake ? <ShieldAlert className="w-20 h-20" /> : <ShieldCheck className="w-20 h-20" />}
                  </div>
                  
                  <div className="flex-1 text-center md:text-left">
                    <h2 className="text-3xl font-bold text-white mb-2">
                      {result.is_fake ? "Alerta de Fake News" : "Notícia Verificada"}
                    </h2>
                    <p className="text-gray-400 text-lg leading-relaxed line-clamp-3">
                      "{result.texto}"
                    </p>
                    <div className="mt-4 flex flex-wrap items-center justify-center md:justify-start gap-3">
                      <span className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-sm font-medium flex items-center gap-2">
                        <Activity className="w-4 h-4 text-blue-400" />
                        Confiança Ensemble: {(result.ensemble_score * 100).toFixed(1)}%
                      </span>
                      <span className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-sm font-medium flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-indigo-400" />
                        {result.nodes} Nós no Grafo
                      </span>
                    </div>
                  </div>
                </div>

                {/* Breakdown dos Modelos (Ensemble) */}
                <div className="mt-10 pt-8 border-t border-white/10">
                  <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-6 flex items-center gap-2">
                    Votação dos Modelos (Ablation)
                  </h3>
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    {result.model_breakdown.map((m: any, idx: number) => (
                      <div key={idx} className="bg-white/5 p-5 rounded-2xl border border-white/5 hover:bg-white/10 transition-colors">
                        <div className="text-xs text-gray-500 mb-1 font-mono uppercase">Peso: {m.weight}x</div>
                        <div className="font-medium text-gray-200 mb-3 text-sm truncate" title={m.model.replace(/_/g, ' ')}>{m.model.replace(/_/g, ' ')}</div>
                        
                        <div className="flex items-end justify-between">
                           <span className={`text-2xl font-bold ${m.prob_fake >= 0.5 ? 'text-red-400' : 'text-emerald-400'}`}>
                             {(m.prob_fake * 100).toFixed(0)}%
                           </span>
                           <span className="text-xs text-gray-500 pb-1">Risco</span>
                        </div>
                        
                        <div className="mt-3 w-full bg-black/50 h-1.5 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full transition-all duration-1000 ${m.prob_fake >= 0.5 ? 'bg-red-500' : 'bg-emerald-500'}`}
                            style={{ width: `${m.prob_fake * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Grafo de Propagação — GNNExplainer */}
                {result.graph_explanation && (
                  <div className="mt-10 pt-8 border-t border-white/10">
                    <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                      Grafo de Propagação — Explicabilidade GNN
                    </h3>
                    <p className="text-xs text-gray-500 mb-5 max-w-lg">
                      Arestas mais espessas e coloridas indicam as interações que mais influenciaram
                      o veredicto do modelo. Gerado via GNNExplainer (PyTorch Geometric).
                    </p>
                    <GraphViz explanation={result.graph_explanation} isFake={result.is_fake} />
                    <GraphLegend isFake={result.is_fake} />
                  </div>
                )}

              </div>
            </div>
          </div>
        )}

        {/* Historico */}
        {history.length > 0 && (
          <div className="mt-16 animate-in fade-in duration-1000 delay-300">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              <Clock className="w-5 h-5 text-gray-400" /> Consultas Recentes
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {history.map(item => (
                <div key={item.id} className="bg-white/5 border border-white/10 p-5 rounded-3xl backdrop-blur-md hover:-translate-y-1 transition-transform cursor-pointer" onClick={() => {setUrl(item.url_bsky || ''); window.scrollTo(0,0);}}>
                  <div className="flex items-center gap-3 mb-3">
                    <div className={`w-3 h-3 rounded-full ${item.heuristica_final >= 0.5 ? 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]' : 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]'}`} />
                    <span className="text-sm font-semibold text-gray-300 uppercase tracking-wider">{item.heuristica_final >= 0.5 ? 'Fake' : 'Real'}</span>
                  </div>
                  <p className="text-sm text-gray-400 line-clamp-2 leading-relaxed">"{item.texto_resumo}"</p>
                  <div className="mt-4 text-xs font-mono text-gray-600 truncate">{item.url_bsky ? item.url_bsky.replace('https://', '') : ''}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* About TCC Section */}
        <div className="mt-24 mb-16 max-w-5xl mx-auto animate-in fade-in duration-1000 delay-500">
          <div className="bg-white/5 border border-white/10 p-8 md:p-12 rounded-[2.5rem] backdrop-blur-2xl">
            <h2 className="text-3xl md:text-4xl font-extrabold text-white mb-6 bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
              Sobre o Projeto de TCC
            </h2>
            
            <div className="prose prose-invert prose-lg max-w-none text-gray-300">
              <p>
                Este portal web (<strong>Truth GNN Analytics</strong>) é a aplicação prática oficial que acompanha o Trabalho de Conclusão de Curso <em>"Detecção de Fake News em Redes Sociais utilizando Graph Neural Networks"</em>.
              </p>
              <p className="mt-4 text-justify">
                Redes sociais contêm uma topologia riquíssima que a leitura simples de texto acaba negligenciando. Nossa abordagem modelou os posts do <strong>Bluesky</strong> e suas interações orgânicas como Árvores de Propagação (Grafos estendidos). Extraímos primeiro as coordenadas semânticas textuais através do BERT-Base e fundimos os tensores com a topologia estrutural mapeada via Redes Convolucionais de Grafos (<strong>GCN</strong>).
              </p>
              
              <h3 className="text-2xl font-bold text-white mt-10 mb-4">Metodologia Ponderada (Ensemble Framework)</h3>
              <p className="text-justify">
                Ao invés de confiar de olhos vendados num único modelo "Mestre", a ferramenta abriga ativamente um motor <em>Ensemble</em> com algoritmos instanciados de forma paralela usando PyTorch e FastAPI. Avaliações profundas (Ablation Study) atestaram que treinar IA's em nichos isolados (e.g., antigo Twitter/UPFD) provoca baixíssima capacidade de rastreio em dados <em>Open-source</em> massivos do Bluesky sem rotulagem fechada, necessitando algoritmos mais "Céticos".
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
                <div className="bg-black/40 p-4 rounded-xl border border-white/5">
                  <span className="text-xs text-gray-400 font-mono uppercase tracking-wider mb-2 block text-center">Performance do UPFD no Bluesky</span>
                  <img src="/images/matriz_upfd_vs_bs_aberto.png" alt="Matriz UPFD" className="w-full rounded-lg shadow-lg opacity-90 hover:opacity-100 transition-opacity" />
                  <p className="text-sm text-gray-400 mt-3 text-center">Altíssimas taxas cruzadas de Falsos Reais.</p>
                </div>
                <div className="bg-black/40 p-4 rounded-xl border border-white/5">
                  <span className="text-xs text-gray-400 font-mono uppercase tracking-wider mb-2 block text-center">Exageradamente Cético (6x Penalidade)</span>
                  <img src="/images/matriz_bs_exa_cetico_vs_bs_aberto.png" alt="Matriz Cética" className="w-full rounded-lg shadow-lg opacity-90 hover:opacity-100 transition-opacity" />
                  <p className="text-sm text-gray-400 mt-3 text-center">Filtro otimizado para expurgo severo de Fakes.</p>
                </div>
              </div>

              <h3 className="text-2xl font-bold text-white mt-12 mb-4">Comportamento Topológico vs Impulsionalidade</h3>
              <p className="text-justify">
                A aplicação expôs estatisticamente (ver gráfico abaixo) que o índice de Vulnerabilidade das Redes de Grafo escala perigosamente quando reduzida a capilaridade da notícia. Ou seja: Publicações Falsas com baixa taxa de comentários (menos de 5 engajamentos) perdem a base de metadados relacionais essencial ao modelo, gerando <em>picos alarmantes de Falsos Reais</em> pela ausência gráfica mitigada no modelo base.
              </p>

              <div className="mt-8 bg-black/40 p-5 rounded-xl border border-white/5 flex flex-col items-center justify-center text-center">
                <span className="text-xs text-gray-400 font-mono uppercase tracking-wider mb-4 block">Distribuição de Erros por Volume Nodal</span>
                <img src="/images/vulnerabilidade_topologia.png" alt="Topologia e Vulnerabilidade" className="w-full max-w-2xl mx-auto rounded-lg shadow-lg opacity-90 hover:opacity-100 transition-opacity" />
                <p className="text-sm mt-4 text-gray-400">Posts menos populares exigem um processador predominantemente Neurolinguístico contra a arquitetura focada puramente em dispersão de grafos.</p>
              </div>

              {/* Equipe / Autores */}
              <div className="mt-16 pt-10 border-t border-white/10">
                <h3 className="text-2xl font-bold text-white mb-8 text-center">Pesquisadores (Trabalho de Conclusão de Curso)</h3>
                <div className="flex flex-wrap justify-center gap-6">
                  
                  <a href="https://cesarsibila.vercel.app/" target="_blank" rel="noopener noreferrer" className="flex items-center gap-4 bg-white/5 hover:bg-white/10 border border-white/10 px-6 py-4 rounded-2xl transition-all hover:scale-105 no-underline">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-tr from-blue-500 to-cyan-400 flex items-center justify-center text-xl font-bold text-white shadow-lg">C</div>
                    <div>
                      <div className="font-bold text-gray-100 m-0 leading-tight">César Sibila</div>
                      <div className="text-xs text-blue-400 font-mono mt-1">Website Pessoal</div>
                    </div>
                  </a>

                  <a href="https://www.linkedin.com/in/andre-messina-506179239/" target="_blank" rel="noopener noreferrer" className="flex items-center gap-4 bg-white/5 hover:bg-white/10 border border-white/10 px-6 py-4 rounded-2xl transition-all hover:scale-105 no-underline">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-400 flex items-center justify-center text-xl font-bold text-white shadow-lg">AM</div>
                    <div>
                      <div className="font-bold text-gray-100 m-0 leading-tight">André Messina</div>
                      <div className="text-xs text-indigo-400 font-mono mt-1">LinkedIn</div>
                    </div>
                  </a>

                  <a href="https://github.com/enzotakida" target="_blank" rel="noopener noreferrer" className="flex items-center gap-4 bg-white/5 hover:bg-white/10 border border-white/10 px-6 py-4 rounded-2xl transition-all hover:scale-105 no-underline">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-tr from-gray-600 to-gray-400 flex items-center justify-center text-xl font-bold text-white shadow-lg">ET</div>
                    <div>
                      <div className="font-bold text-gray-100 m-0 leading-tight">Enzo Takida</div>
                      <div className="text-xs text-gray-400 font-mono mt-1">GitHub</div>
                    </div>
                  </a>

                </div>
              </div>

            </div>
          </div>
        </div>

      </div>
    </main>
  );
}
