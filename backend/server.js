
require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const mysql = require('mysql2/promise');

const fetch = require('node-fetch');
const { spawn } = require('child_process');

const app = express();
app.use(bodyParser.json());

const DB = {
  host: process.env.DB_HOST || 'localhost',
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASS || '123456',
  database: process.env.DB_NAME || 'internship_chat1'
};

const AGENT_URL = process.env.AGENT_URL || 'http://localhost:8000/run_agent';
const GEMINI_EMBED_URL = process.env.GEMINI_EMBED_URL;
const TRIVY_PATH = process.env.TRIVY_PATH || 'trivy';

// DB pool helper
const pool = mysql.createPool(DB);

async function saveMessage(userId, role, content, metadata = {}) {
  const conn = await pool.getConnection();
  try {
    await conn.execute(
      'INSERT INTO chat_messages (user_id, role, content, metadata) VALUES (?,?,?,?)',
      [userId, role, content, JSON.stringify(metadata)]
    );
  } finally {
    conn.release();
  }
}

// Basic embedding: calls GEMINI_EMBED_URL (mock or real)
async function getEmbedding(text) {
  // POST { input: "..." } => { embedding: [ ... ] }
  const resp = await fetch(GEMINI_EMBED_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: text })
  });
  if (!resp.ok) {
    const t = await resp.text();
    throw new Error('Embedding call failed: ' + resp.status + ' ' + t);
  }
  const j = await resp.json();
  return j.embedding; // should be array of floats
}

// Cosine similarity
function cosine(a,b){
  let dot=0, na=0, nb=0;
  for(let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i];}
  return dot/(Math.sqrt(na)*Math.sqrt(nb)+1e-12);
}

// RAG retrieve
async function ragRetrieve(queryEmb, topK=3){
  const conn = await pool.getConnection();
  try {
    const [rows] = await conn.query('SELECT id,title,content,embedding FROM documents');
    const docs = rows.map(r => ({id:r.id, title:r.title, content:r.content, emb: JSON.parse(r.embedding)}));
    docs.forEach(d => d.score = cosine(queryEmb, d.emb));
    docs.sort((a,b)=> b.score - a.score);
    return docs.slice(0, topK);
  } finally {
    conn.release();
  }
}

// Save sys prompt
app.post('/api/prompt', async (req,res) => {
  try {
    const { name='default', prompt } = req.body;
    const conn = await pool.getConnection();
    await conn.execute('INSERT INTO system_prompts (name, prompt) VALUES (?,?)', [name, prompt]);
    conn.release();
    return res.json({ ok: true });
  } catch(e){ console.error(e); return res.status(500).json({error: e.message}); }
});

// Insert document (for RAG)
app.post('/api/doc', async (req,res) => {
  try {
    const { title, content } = req.body;
    const emb = await getEmbedding(content);
    const conn = await pool.getConnection();
    await conn.execute('INSERT INTO documents (title, content, embedding) VALUES (?,?,?)',
      [title, content, JSON.stringify(emb)]);
    conn.release();
    return res.json({ ok:true });
  } catch(e){ console.error(e); return res.status(500).json({error: e.message}); }
});

// List chat history by user
app.get('/api/history/:userId', async (req,res) => {
  try {
    const uid = Number(req.params.userId || 1);
    const conn = await pool.getConnection();
    const [rows] = await conn.execute('SELECT id,user_id,role,content,metadata,created_at FROM chat_messages WHERE user_id=? ORDER BY created_at', [uid]);
    conn.release();
    return res.json(rows);
  } catch(e){ console.error(e); return res.status(500).json({error: e.message}); }
});

// Trivy scan endpoint (spawns trivy). target is directory or image.
app.post('/api/scan', async (req,res) => {
  try {
    const { target='.' } = req.body;
    const proc = spawn(TRIVY_PATH, ['--quiet', 'fs', target]);
    let out = '', errout = '';
    proc.stdout.on('data', d => out += d.toString());
    proc.stderr.on('data', d => errout += d.toString());
    proc.on('close', code => res.json({ code, out, err: errout }));
  } catch(e){ console.error(e); return res.status(500).json({error: e.message}); }
});

// Chat endpoint: main flow
app.post('/api/chat', async (req,res) => {
  try {
    const userId = req.body.userId || 1;
    const message = req.body.message;
    if (!message) return res.status(400).json({error:'message required'});

    // 1) persist user message
    await saveMessage(userId, 'user', message);

    // 2) embedding + RAG
    const emb = await getEmbedding(message);
    const docs = await ragRetrieve(emb, 3);

    // 3) load latest system prompt
    const conn = await pool.getConnection();
    const [prompts] = await conn.execute('SELECT prompt FROM system_prompts ORDER BY created_at DESC LIMIT 1');
    conn.release();
    const system_prompt = (prompts && prompts.length) ? prompts[0].prompt : "You are a helpful assistant.";

    // 4) call agent service (Python) with system prompt + context
    const agentResp = await fetch(AGENT_URL, {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({
        userId, message,
        system_prompt,
        context: docs.map(d => ({ title:d.title, content:d.content, score:d.score }))
      })
    });
    if (!agentResp.ok) {
      const t = await agentResp.text();
      throw new Error(`Agent call failed ${agentResp.status} ${t}`);
    }
    const agentJson = await agentResp.json();
    const assistantReply = agentJson.reply || "(no reply)";

    // 5) persist assistant reply
    await saveMessage(userId, 'assistant', assistantReply, { tools: agentJson.tools || [] });

    return res.json({ reply: assistantReply, tools: agentJson.tools || [] });
  } catch(e){ console.error(e); return res.status(500).json({error: e.message}); }
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, ()=> console.log(`Node backend listening on ${PORT}`));
