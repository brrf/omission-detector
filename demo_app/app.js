(() => {
    const el = (id) => document.getElementById(id);
    const runListEl = el('runList');
    const encounterView = el('encounterView');
    const emptyState = el('emptyState');
    const runTitle = el('runTitle');
    const omissionList = el('omissionList');
    const hpiOriginal = el('hpiOriginal');
    const hpiRevised = el('hpiRevised');
    const generateBtn = el('generateBtn');
    const sendBtn = el('sendBtn');
    const statusEl = el('status');
    const selectAllBtn = el('selectAllBtn');
    const deselectAllBtn = el('deselectAllBtn');
    const searchInput = el('searchInput');
    const metaPaths = el('metaPaths');
  
    let DATA = { runs: [], meta: {} };
    let current = null;
  
    function setStatus(msg) {
      statusEl.textContent = msg || '';
    }
  
    function boldMarkdownToHtml(text){
      // minimal rendering for **bold**; leave other markdown as-is
      if(!text) return '';
      return text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    }
  
    function renderRuns(filter=''){
      runListEl.innerHTML = '';
      const filterLc = (filter||'').trim().toLowerCase();
      DATA.runs
        .filter(r => r.run_id.toLowerCase().includes(filterLc))
        .sort((a,b) => a.run_id.localeCompare(b.run_id))
        .forEach(r => {
          const li = document.createElement('li');
          li.className = 'run-item';
          li.innerHTML = `<div class="id">${r.run_id}</div>
                          <div class="count">${(r.omissions||[]).length} omissions</div>`;
          li.onclick = () => openRun(r.run_id);
          runListEl.appendChild(li);
        });
    }
  
    function openRun(runId){
      const run = DATA.runs.find(r => r.run_id === runId);
      if(!run) return;
      current = JSON.parse(JSON.stringify(run)); // clone to allow UI state
      runTitle.textContent = `Run: ${run.run_id}`;
      hpiOriginal.textContent = run.hpi || '(No HPI found for this encounter)';
      hpiRevised.innerHTML = '';
  
      // omissions list
      omissionList.innerHTML = '';
      (current.omissions || []).forEach((o, idx) => {
        o.__selected = true; // default selected
        const li = document.createElement('li');
        li.className = 'omit-item';
        const label = [
          o.code ? `Code: ${o.code}` : null,
          o.polarity ? `Polarity: ${o.polarity}` : null,
          o.time_scope ? `Time: ${o.time_scope}` : null,
          o.priority ? `Priority: ${o.priority}` : null,
          (o.materiality != null) ? `Materiality: ${o.materiality}` : null
        ].filter(Boolean).join(' • ');
  
        li.innerHTML = `
          <input type="checkbox" checked data-idx="${idx}" />
          <div>
            <div class="meta">${label}</div>
            <div class="val">${o.value || '(no value)'}</div>
            ${o.evidence_text ? `<div class="meta">Evidence: “${o.evidence_text}”</div>` : ''}
          </div>
        `;
        const cb = li.querySelector('input[type="checkbox"]');
        cb.addEventListener('change', (e) => {
          const i = Number(e.target.getAttribute('data-idx'));
          current.omissions[i].__selected = e.target.checked;
        });
        omissionList.appendChild(li);
      });
  
      emptyState.classList.add('hidden');
      encounterView.classList.remove('hidden');
      setStatus('');
    }
  
    async function fetchData(){
      const res = await fetch('/api/data');
      const json = await res.json();
      if(json.error){
        throw new Error(json.error);
      }
      DATA = json;
      const omitPath = (json.meta && (json.meta.omissions_jsonl || json.meta.omissions_yaml)) || 'out/omissions.jsonl';
        metaPaths.innerHTML = `
        <span>Loaded: <code>${omitPath}</code></span>
        &nbsp;·&nbsp;
        <span>HPI CSV: <code>${json.meta.hpi_csv}</code></span>
        `;
      renderRuns();
    }
  
    async function generateNote(){
      if(!current) return;
      setStatus('Generating with gpt-4o-mini…');
      generateBtn.disabled = true;
  
      const selected = (current.omissions || []).filter(o => o.__selected);
      if(selected.length === 0){
        setStatus('Nothing selected.');
        generateBtn.disabled = false;
        return;
      }
  
      // payload trimmed to what model needs
      const payload = {
        run_id: current.run_id,
        original_hpi: current.hpi || '',
        selected_omissions: selected.map(o => ({
          id: o.id,
          code: o.code,
          value: o.value,
          polarity: o.polarity,
          time_scope: o.time_scope,
          evidence_text: o.evidence_text,
          priority: o.priority,
          materiality: o.materiality
        })),
      };
  
      try{
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const json = await res.json();
        if(json.error){
          throw new Error(json.error);
        }
        const text = json.new_hpi || '';
        hpiRevised.innerHTML = boldMarkdownToHtml(text);
        setStatus('Done.');
      } catch (e){
        console.error(e);
        setStatus('Error generating note: ' + e.message);
      } finally{
        generateBtn.disabled = false;
      }
    }
  
    // Wire up events
    generateBtn.addEventListener('click', generateNote);
    sendBtn.addEventListener('click', () => {
      setStatus('Sent to clinician (stub).');
    });
    selectAllBtn.addEventListener('click', () => {
      if(!current) return;
      (current.omissions||[]).forEach(o => o.__selected = true);
      omissionList.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
    });
    deselectAllBtn.addEventListener('click', () => {
      if(!current) return;
      (current.omissions||[]).forEach(o => o.__selected = false);
      omissionList.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
    });
    searchInput.addEventListener('input', (e) => {
      renderRuns(e.target.value);
    });
  
    // init
    fetchData().catch(err => {
      console.error(err);
      setStatus('Failed to load data: ' + err.message);
    });
  })();
  