let db = null;
let SQL = null;
let currentFileBytes = null;

const fileInput = document.getElementById('fileInput');
const downloadBtn = document.getElementById('downloadBtn');
const sourceFilter = document.getElementById('sourceFilter');
const docChunksContainer = document.getElementById('docChunksContainer');

async function loadSqlJsModule() {
  if (SQL) return SQL;
  // the sql.js script exposes a global initSqlJs function; call that rather than recursing
  const initFn = globalThis.initSqlJs || window.initSqlJs;
  if (typeof initFn !== 'function') {
    throw new Error('sql.js initSqlJs not found; ensure sql-wasm.js is loaded');
  }
  SQL = await initFn({ locateFile: file => 'https://cdn.jsdelivr.net/npm/sql.js@1.8.0/dist/sql-wasm.wasm' });
  return SQL;
}

function bytesToArrayBuffer(bytes) {
  return bytes.buffer ? bytes.buffer : new Uint8Array(bytes).buffer;
}

function arrayBufferToUint8(a) {
  return new Uint8Array(a);
}

async function loadDatabaseFromFile(file) {
  const buf = await file.arrayBuffer();
  currentFileBytes = new Uint8Array(buf);
  const SQLmod = await loadSqlJsModule();
  db = new SQLmod.Database(currentFileBytes);
  // populate document-centric UI
  loadDocumentsList();
  populateSourceFilter();
  populateTagSuggestions();
  downloadBtn.disabled = false;
}
// removed raw table viewer functions; this tool is document-centric only

function downloadDatabase() {
  const data = db.export();
  const blob = new Blob([data], {type: 'application/octet-stream'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'documents-modified.db'; a.click();
  URL.revokeObjectURL(url);
}

fileInput.addEventListener('change', (ev) => {
  const f = ev.target.files && ev.target.files[0];
  if (!f) return;
  loadDatabaseFromFile(f);
});

downloadBtn.addEventListener('click', downloadDatabase);

// ensure sql.js loads early
loadSqlJsModule().catch(err => alert('Failed to load SQL WASM: '+err.message));

// --- Document-centric UI ---
const docListEl = document.getElementById('docTable').querySelector('tbody');
const docFilter = document.getElementById('docFilter');

function loadDocumentsList() {
  // load basic document rows: doc_id, shortname, date
  try {
  const res = db.exec("SELECT doc_id, shortname, source, date FROM documents ORDER BY doc_id DESC");
    if (!res || res.length === 0) return;
    const rows = res[0].values;
    renderDocList(rows);
  } catch (e) {
    console.warn('No documents table or failed to query:', e.message);
  }
}

function formatDate(raw) {
  if (!raw) return '';
  // Try Date parse first
  const tryDate = new Date(raw);
  if (!isNaN(tryDate)) {
    const day = tryDate.getDate();
    const month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][tryDate.getMonth()];
    const year = tryDate.getFullYear();
    return `${day} ${month} ${year}`;
  }
  // fallback: attempt to extract YYYY-MM-DD
  const m = String(raw).match(/(\d{4})-(\d{2})-(\d{2})/);
  if (m) {
    const y = m[1], mo = parseInt(m[2],10)-1, day = parseInt(m[3],10);
    const month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][mo];
    return `${day} ${month} ${y}`;
  }
  return String(raw);
}

function renderDocList(rows) {
  docListEl.innerHTML = '';
  for (const r of rows) {
    const id = r[0]; const shortname = r[1] || ''; const source = r[2] || ''; const date = r[3] || '';
    const tr = document.createElement('tr');
    tr.dataset.docId = id;
    tr.style.cursor = 'pointer';
  tr.innerHTML = `<td style="padding:8px">${id}</td><td style="padding:8px">${shortname}</td><td style="padding:8px">${source}</td><td style="padding:8px">${formatDate(date)}</td>`;
    tr.onclick = () => {
      // clear previous selection
      for (const rr of Array.from(docListEl.querySelectorAll('tr'))) rr.classList.remove('selected');
      tr.classList.add('selected');
      openDocument(id);
    };
    docListEl.appendChild(tr);
  }

  docFilter.oninput = () => {
    const q = docFilter.value.trim().toLowerCase();
    for (const row of Array.from(docListEl.querySelectorAll('tr'))) {
      const txt = row.textContent.toLowerCase();
      row.style.display = q === '' || txt.includes(q) ? '' : 'none';
    }
  };
}

function openDocument(docId) {
  docChunksContainer.innerHTML = '';
  // fetch document rows joined with chunks and tags
  // documents -> docs_chunks -> chunks ; tags are in tags table
  try {
    const stmt = db.prepare(`SELECT d.doc_id, d.shortname, d.source, d.date,
        c.chunk_id, c.chunk_text, t.tag, dc.chunk_order
      FROM documents d
      LEFT JOIN docs_chunks dc ON d.doc_id = dc.doc_id
      LEFT JOIN chunks c ON dc.chunk_id = c.chunk_id
      LEFT JOIN tags t ON t.chunk_id = c.chunk_id
      WHERE d.doc_id = ?
      ORDER BY dc.chunk_order ASC`);
    stmt.bind([docId]);
    const cols = stmt.getColumnNames();
    const rows = [];
    while (stmt.step()) rows.push(stmt.get());
    stmt.free();

    if (rows.length === 0) {
      docChunksContainer.textContent = 'No rows for this document';
      return;
    }

    // group chunks with tags; track chunk_order for explicit sort
    const docMeta = { doc_id: rows[0][0], shortname: rows[0][1], source: rows[0][2], date: rows[0][3] };
    const chunksMap = new Map();
    for (const r of rows) {
      const chunk_id = r[4];
      const chunk_text = r[5];
      const tag = r[6];
      const chunk_order = r[7];
      if (!chunksMap.has(chunk_id)) {
        chunksMap.set(chunk_id, { chunk_id, chunk_text, tags: new Set(), chunk_order });
      } else {
        // update text/order if needed (in case of duplicates from tags join)
        const existing = chunksMap.get(chunk_id);
        if (existing.chunk_text !== chunk_text) existing.chunk_text = chunk_text;
        if (existing.chunk_order !== chunk_order) existing.chunk_order = chunk_order;
      }
      if (tag) chunksMap.get(chunk_id).tags.add(tag);
    }
    const chunksArr = Array.from(chunksMap.values()).sort((a,b) => {
      // ensure numeric comparison; handle null/undefined by pushing to end
      const ao = a.chunk_order; const bo = b.chunk_order;
      if (ao == null && bo == null) return 0;
      if (ao == null) return 1;
      if (bo == null) return -1;
      return ao - bo;
    });

    const header = document.createElement('div');
    header.innerHTML = `<h3>Document ${docMeta.doc_id} — ${docMeta.shortname}</h3><div class="muted">${docMeta.source} — ${docMeta.date}</div>`;
    docChunksContainer.appendChild(header);

  // Build a two-column table: chunk text | tags
  const tbl = document.createElement('table');
  tbl.className = 'chunk-table';
    tbl.style.width = '100%';
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  const th1 = document.createElement('th'); th1.textContent = 'Chunk text';
  const th2 = document.createElement('th'); th2.textContent = 'Tags';
  hr.appendChild(th1); hr.appendChild(th2); thead.appendChild(hr); tbl.appendChild(thead);
  const colgroup = document.createElement('colgroup');
  const col1 = document.createElement('col'); col1.className='chunk-col';
  const col2 = document.createElement('col'); col2.className='tag-col';
  colgroup.appendChild(col1); colgroup.appendChild(col2); tbl.appendChild(colgroup);
  const tbody = document.createElement('tbody');

    for (const ch of chunksArr) {
      const tr = document.createElement('tr');
      const tdText = document.createElement('td');
      const ta = document.createElement('textarea'); ta.style.width='100%'; ta.rows=5; ta.value = ch.chunk_text || '';
      tdText.appendChild(ta);
      const saveBtn = document.createElement('button'); saveBtn.textContent='Save'; saveBtn.className='btn';
      saveBtn.onclick = () => {
        try {
          const st = db.prepare('UPDATE chunks SET chunk_text = ? WHERE chunk_id = ?');
          st.run([ta.value, ch.chunk_id]);
          st.free();
          alert('Saved chunk');
        } catch (e) { alert('Error: '+e.message); }
      };
      tdText.appendChild(saveBtn);

      const tdTags = document.createElement('td');
      // render tags as badges with remove anchors
      function renderTagsForChunk(ch, container) {
        container.innerHTML = '';
        for (const t of Array.from(ch.tags)) {
          const span = document.createElement('span'); span.className = 'tag-badge';
          const label = document.createElement('span'); label.textContent = t;
          const rem = document.createElement('span'); rem.className = 'remove'; rem.textContent = '×';
          rem.title = 'Remove tag';
          rem.onclick = () => {
            if (!confirm(`Remove tag "${t}" from chunk ${ch.chunk_id}?`)) return;
            try {
              const st = db.prepare('DELETE FROM tags WHERE chunk_id = ? AND tag = ?');
              st.run([ch.chunk_id, t]); st.free();
              ch.tags.delete(t);
              populateTagSuggestions();
              renderTagsForChunk(ch, container);
            } catch (e) { alert('Error removing tag: '+e.message); }
          };
          span.appendChild(label); span.appendChild(rem); container.appendChild(span);
        }
        const add = document.createElement('span'); add.className = 'tag-add'; add.textContent = '+ add';
        add.onclick = () => {
          // replace add pill with inline input + save/cancel
          const input = document.createElement('input'); input.setAttribute('list', 'tagSuggestions'); input.placeholder='tag';
          input.style.marginRight='6px';
          const save = document.createElement('button'); save.textContent='Save'; save.className='btn';
          const cancel = document.createElement('button'); cancel.textContent='Cancel'; cancel.className='btn';
          container.appendChild(input); container.appendChild(save); container.appendChild(cancel);
          input.focus();
          save.onclick = () => {
            const val = input.value && input.value.trim(); if (!val) { alert('Empty tag'); return; }
            try {
              const st = db.prepare('INSERT INTO tags (chunk_id, tag) VALUES (?, ?)');
              st.run([ch.chunk_id, val]); st.free();
              ch.tags.add(val);
              populateTagSuggestions();
              renderTagsForChunk(ch, container);
            } catch (e) { alert('Error adding tag: '+e.message); }
          };
          cancel.onclick = () => { renderTagsForChunk(ch, container); };
        };
        container.appendChild(add);
      }
      renderTagsForChunk(ch, tdTags);

      tr.appendChild(tdText); tr.appendChild(tdTags); tbody.appendChild(tr);
    }

    tbl.appendChild(tbody);
    docChunksContainer.appendChild(tbl);

  } catch (e) {
    docChunksContainer.textContent = 'Failed to load document: ' + e.message;
  }
}

function populateSourceFilter() {
  // gather distinct sources from documents
  try {
    const res = db.exec("SELECT DISTINCT source FROM documents ORDER BY source");
    if (!res || res.length === 0) return;
    const values = res[0].values.map(r => r[0] || '');
    sourceFilter.innerHTML = '<option value="">(all)</option>' + values.map(v => `<option value="${v}">${v}</option>`).join('');
    sourceFilter.onchange = () => {
      // reload document list filtered by source
      const s = sourceFilter.value;
      if (!s) {
        loadDocumentsList();
        return;
      }
      try {
  const stmt = db.prepare('SELECT doc_id, shortname, source, date FROM documents WHERE source = ? ORDER BY date DESC');
        stmt.bind([s]);
        const rows = [];
        while (stmt.step()) rows.push(stmt.get());
        stmt.free();
        renderDocList(rows);
      } catch (e) { console.warn(e.message); }
    };
  } catch (e) { console.warn('populateSourceFilter failed', e.message); }
}

function populateTagSuggestions() {
  try {
    const res = db.exec('SELECT DISTINCT tag FROM tags ORDER BY tag');
    if (!res || res.length === 0) return;
    const values = res[0].values.map(r => r[0]);
    const dl = document.getElementById('tagSuggestions');
    dl.innerHTML = values.map(v => `<option value="${v}"></option>`).join('');
  } catch (e) { console.warn('populateTagSuggestions failed', e.message); }
}
