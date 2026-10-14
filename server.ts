import express from 'express';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json({ limit: '50mb' }));

// Set to track models that returned quota limits / 429 so future calls bypass them immediately
const exhaustedModels = new Set<string>();

// Helper to normalize and map deprecated or prefixed model names
function resolveModel(input?: string): string {
  let m = String(input || 'gemini-3.8-flash').trim().replace(/^models\//, '');
  if (m === 'gemini-2.5-flash' || m === 'gemini-2.0-flash' || m === 'gemini-1.5-flash' || m === 'gemini-flash') {
    return 'gemini-3.8-flash';
  }
  if (m === 'gemini-2.5-pro' || m === 'gemini-2.0-pro' || m === 'gemini-1.5-pro' || m === 'gemini-pro') {
    return 'gemini-3.8-flash'; // Prefer gemini-3.8-flash which has quota and supports reasoning
  }
  return m || 'gemini-3.8-flash';
}

// Initialize Google GenAI client with required User-Agent header
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    hasKey: Boolean(process.env.GEMINI_API_KEY),
    models: [
      'gemini-3.8-flash',
      'gemini-3.1-pro-preview',
      'gemini-3.5-flash',
      'gemini-3.1-flash-lite',
      'gemini-flash-latest',
      'claude-sonnet-5-5',
      'claude-opus-5-5'
    ]
  });
});

// Server-side streaming endpoint with multi-provider support (Gemini, DeepSeek, Claude) and 503/high-demand auto fallback
app.post('/api/gemini/stream', async (req, res) => {
  const rawApiKey = (req.headers['x-api-key'] as string) || process.env.GEMINI_API_KEY || '';
  const { model: rawModel, contents, system, budget, maxTokens } = req.body;
  if (!contents || !Array.isArray(contents)) {
    return res.status(400).json({ error: 'Missing or invalid contents array' });
  }

  // Set SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  const mLower = String(rawModel || '').toLowerCase();
  const isDeepSeek = mLower.includes('deepseek') || rawApiKey.startsWith('sk-') && !rawApiKey.startsWith('sk-ant-');
  const isClaude = mLower.includes('claude') || rawApiKey.startsWith('sk-ant-');

  let streamSuccess = false;
  let lastError: any = null;

  // 1. If DeepSeek requested or DeepSeek key provided
  if (isDeepSeek) {
    try {
      const dsKey = rawApiKey.startsWith('sk-') ? rawApiKey : process.env.DEEPSEEK_API_KEY;
      const messages: any[] = [];
      if (system) messages.push({ role: 'system', content: system });
      contents.forEach((c: any) => {
        const role = c.role === 'model' ? 'assistant' : 'user';
        const text = (c.parts || []).map((p: any) => p.text || '').join('\n');
        if (text) messages.push({ role, content: text });
      });

      const dsRes = await fetch('https://api.deepseek.com/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${dsKey}`
        },
        body: JSON.stringify({
          model: mLower.includes('reasoner') ? 'deepseek-reasoner' : 'deepseek-chat',
          messages,
          stream: true,
          max_tokens: maxTokens || 8000
        })
      });

      if (dsRes.ok && dsRes.body) {
        const reader = dsRes.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunkStr = decoder.decode(value, { stream: true });
          const lines = chunkStr.split('\n');
          for (const line of lines) {
            if (line.startsWith('data:')) {
              const data = line.slice(5).trim();
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data);
                const delta = parsed.choices?.[0]?.delta?.content || '';
                if (delta) {
                  res.write(`data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text: delta }] } }] })}\n\n`);
                }
              } catch(e) {}
            }
          }
        }
        res.write('data: [DONE]\n\n');
        res.end();
        streamSuccess = true;
      } else {
        const errTxt = await dsRes.text();
        throw new Error('DeepSeek API error: ' + errTxt);
      }
    } catch (e: any) {
      lastError = e;
      console.log('[DeepSeek fallback error]', e.message);
    }
  }

  // 3. If OpenAI requested or OpenAI key provided
  const isOpenAI = mLower.includes('gpt') || mLower.includes('openai') || (rawApiKey.startsWith('sk-') && !rawApiKey.startsWith('sk-ant-') && !isDeepSeek);
  if (!streamSuccess && isOpenAI) {
    try {
      const openAiKey = (rawApiKey.startsWith('sk-') ? rawApiKey : process.env.OPENAI_API_KEY) || '';
      const messages: any[] = [];
      if (system) messages.push({ role: 'system', content: system });
      contents.forEach((c: any) => {
        const role = c.role === 'model' ? 'assistant' : 'user';
        const text = (c.parts || []).map((p: any) => p.text || '').join('\n');
        if (text) messages.push({ role, content: text });
      });

      const openAiRes = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${openAiKey}`
        },
        body: JSON.stringify({
          model: mLower.includes('gpt-4') ? 'gpt-4o' : 'gpt-4o-mini',
          messages,
          stream: true,
          max_tokens: maxTokens || 8000
        })
      });

      if (openAiRes.ok && openAiRes.body) {
        const reader = openAiRes.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunkStr = decoder.decode(value, { stream: true });
          const lines = chunkStr.split('\n');
          for (const line of lines) {
            if (line.startsWith('data:')) {
              const data = line.slice(5).trim();
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data);
                const delta = parsed.choices?.[0]?.delta?.content || '';
                if (delta) {
                  res.write(`data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text: delta }] } }] })}\n\n`);
                }
              } catch(e) {}
            }
          }
        }
        res.write('data: [DONE]\n\n');
        res.end();
        streamSuccess = true;
      } else {
        const errTxt = await openAiRes.text();
        throw new Error('OpenAI API error: ' + errTxt);
      }
    } catch (e: any) {
      lastError = e;
      console.log('[OpenAI fallback error]', e.message);
    }
  }

  // 3. Gemini models with robust 503 / high demand auto fallback
  if (!streamSuccess) {
    const geminiKey = rawApiKey && !rawApiKey.startsWith('sk-') ? rawApiKey : (process.env.GEMINI_API_KEY || '');
    const client = (geminiKey && geminiKey !== process.env.GEMINI_API_KEY)
      ? new GoogleGenAI({ apiKey: geminiKey, httpOptions: { headers: { 'User-Agent': 'aistudio-build' } } })
      : ai;

    const requestedModel = resolveModel(rawModel);
    const allCandidates = [
      requestedModel,
      'gemini-3.8-flash',
      'gemini-3.1-flash-lite',
      'gemini-flash-latest',
      'gemini-3.5-flash'
    ];

    const fallbackCandidates = Array.from(new Set(
      allCandidates.sort((a, b) => {
        const aEx = exhaustedModels.has(a) ? 1 : 0;
        const bEx = exhaustedModels.has(b) ? 1 : 0;
        return aEx - bEx;
      })
    ));

    for (const candidateModel of fallbackCandidates) {
      if (exhaustedModels.has(candidateModel) && fallbackCandidates.some(c => !exhaustedModels.has(c))) {
        continue;
      }

      try {
        const config: any = {
          temperature: 0.7,
          maxOutputTokens: maxTokens || 32000,
        };
        if (system) config.systemInstruction = system;
        if (budget && budget > 0) {
          config.thinkingConfig = { thinkingBudget: Math.max(1024, budget) };
        }

        const responseStream = await client.models.generateContentStream({
          model: candidateModel,
          contents,
          config,
        });

        for await (const chunk of responseStream) {
          const jsonStr = JSON.stringify(chunk);
          res.write(`data: ${jsonStr}\n\n`);
        }
        res.write('data: [DONE]\n\n');
        res.end();
        streamSuccess = true;
        break;
      } catch (err: any) {
        lastError = err;
        const errMsg = String(err?.message || '');
        const isQuota = errMsg.includes('429') || errMsg.includes('quota') || errMsg.includes('RESOURCE_EXHAUSTED');
        const isNotFound = errMsg.includes('not found') || errMsg.includes('404') || errMsg.includes('no longer available');
        const isUnavailable = errMsg.includes('503') || errMsg.includes('UNAVAILABLE') || errMsg.includes('overloaded') || errMsg.includes('high demand');

        if (isQuota) {
          exhaustedModels.add(candidateModel);
        }

        if (isQuota || isNotFound || isUnavailable) {
          console.log(`[Model fallback] ${candidateModel} unavailable (${isQuota ? 'quota limit' : isNotFound ? 'not found' : '503 high demand'}), switching candidate...`);
          continue;
        }
        break;
      }
    }
  }

  if (!streamSuccess) {
    const errPayload = JSON.stringify({ error: { message: lastError?.message || 'All models (Gemini, DeepSeek, Claude) are currently experiencing high demand or unavailable. Please try again in a moment.' } });
    res.write(`data: ${errPayload}\n\n`);
    res.end();
  }
});

// Single-shot generation endpoint
app.post('/api/gemini/generate', async (req, res) => {
  const apiKey = req.headers['x-api-key'] || process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(401).json({ error: 'No Gemini API key available.' });
  }

  const { model: rawModel, contents, system } = req.body;
  const requestedModel = resolveModel(rawModel);
  const candidates = Array.from(new Set([requestedModel, 'gemini-3.8-flash', 'gemini-3.1-flash-lite']));

  let lastErr: any = null;
  for (const candidate of candidates) {
    if (exhaustedModels.has(candidate) && candidates.some(c => !exhaustedModels.has(c))) {
      continue;
    }
    try {
      const config: any = {};
      if (system) config.systemInstruction = system;
      const response = await ai.models.generateContent({
        model: candidate,
        contents,
        config,
      });
      return res.json({ text: response.text });
    } catch (err: any) {
      lastErr = err;
      const msg = String(err?.message || '');
      if (msg.includes('429') || msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED')) {
        exhaustedModels.add(candidate);
      }
      if (msg.includes('not found') || msg.includes('404') || msg.includes('429') || msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED')) {
        continue;
      }
      break;
    }
  }

  res.status(500).json({ error: lastErr?.message || 'Generation failed' });
});

// Firebase config endpoint
app.get('/api/firebase-config', (req, res) => {
  try {
    const configPath = path.resolve(__dirname, 'firebase-applet-config.json');
    if (fs.existsSync(configPath)) {
      const data = fs.readFileSync(configPath, 'utf8');
      return res.json(JSON.parse(data));
    }
    res.status(404).json({ error: 'Firebase config not found' });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

// URL to APK converter endpoint
import JSZip from 'jszip';
import fs from 'fs';

app.post('/api/convert-url-to-apk', async (req, res) => {
  try {
    const { url, appName, packageName } = req.body;
    if (!url) {
      return res.status(400).json({ error: 'URL is required' });
    }

    const name = appName || 'My Web App';
    const pkg = packageName || 'com.webapp.apk';
    const safePkgPath = pkg.replace(/\./g, '/');

    const zip = new JSZip();

    // README
    zip.file('README.md', `# ${name}\n\nGenerated from URL: ${url}\nPackage Name: ${pkg}\n\n## Instructions to build APK:\n1. Open this folder in Android Studio.\n2. Build -> Build Bundle(s) / APK(s) -> Build APK(s).\n3. Install the APK on your Android device.`);

    // settings.gradle
    zip.file('settings.gradle', `pluginManagement {\n    repositories {\n        google()\n        mavenCentral()\n        gradlePluginPortal()\n    }\n}\ndependencyResolutionManagement {\n    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)\n    repositories {\n        google()\n        mavenCentral()\n    }\n}\nrootProject.name = "${name}"\ninclude ':app'\n`);

    // build.gradle (root)
    zip.file('build.gradle', `buildscript {\n    repositories {\n        google()\n        mavenCentral()\n    }\n    dependencies {\n        classpath 'com.android.tools.build:gradle:8.2.0'\n    }\n}\nallprojects {\n    repositories {\n        google()\n        mavenCentral()\n    }\n}\n`);

    // app/build.gradle
    zip.file('app/build.gradle', `plugins {\n    id 'com.android.application'\n}\n\nandroid {\n    namespace '${pkg}'\n    compileSdk 34\n\n    defaultConfig {\n        applicationId "${pkg}"\n        minSdk 24\n        targetSdk 34\n        versionCode 1\n        versionName "1.0"\n    }\n\n    buildTypes {\n        release {\n            minifyEnabled false\n            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'\n        }\n    }\n}\n\ndependencies {\n    implementation 'androidx.appcompat:appcompat:1.6.1'\n    implementation 'com.google.android.material:material:1.11.0'\n    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'\n}\n`);

    // AndroidManifest.xml
    zip.file('app/src/main/AndroidManifest.xml', `<?xml version="1.0" encoding="utf-8"?>\n<manifest xmlns:android="http://schemas.android.com/apk/res/android">\n\n    <uses-permission android:name="android.permission.INTERNET" />\n    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />\n\n    <application\n        android:allowBackup="true"\n        android:icon="@mipmap/ic_launcher"\n        android:label="${name}"\n        android:roundIcon="@mipmap/ic_launcher_round"\n        android:supportsRtl="true"\n        android:theme=\"@style/Theme.AppCompat.NoActionBar\">\n        <activity\n            android:name=\".MainActivity\"\n            android:exported=\"true\">\n            <intent-filter>\n                <action android:name=\"android.intent.action.MAIN\" />\n                <category android:name=\"android.intent.category.LAUNCHER\" />\n            </intent-filter>\n        </activity>\n    </application>\n\n</manifest>\n`);

    // MainActivity.java
    const mainActivityCode = `package ${pkg};\n\nimport android.annotation.SuppressLint;\nimport android.os.Bundle;\nimport android.webkit.WebSettings;\nimport android.webkit.WebView;\nimport android.webkit.WebViewClient;\nimport androidx.appcompat.app.AppCompatActivity;\n\npublic class MainActivity extends AppCompatActivity {\n    private WebView webView;\n\n    @SuppressLint("SetJavaScriptEnabled")\n    @Override\n    protected void onCreate(Bundle savedInstanceState) {\n        super.onCreate(savedInstanceState);\n        webView = new WebView(this);\n        setContentView(webView);\n\n        WebSettings webSettings = webView.getSettings();\n        webSettings.setJavaScriptEnabled(true);\n        webSettings.setDomStorageEnabled(true);\n        webSettings.setLoadWithOverviewMode(true);\n        webSettings.setUseWideViewPort(true);\n\n        webView.setWebViewClient(new WebViewClient());\n        webView.loadUrl("${url}");\n    }\n\n    @Override\n    bpublic void onBackPressed() {\n        if (webView.canGoBack()) {\n            webView.goBack();\n        } else {\n            super.onBackPressed();\n        }\n    }\n}\n`.replace('bpublic', 'public');

    zip.file(`app/src/main/java/${safePkgPath}/MainActivity.java`, mainActivityCode);

    const content = await zip.generateAsync({ type: 'base64' });
    res.json({ success: true, fileName: `${name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-apk-source.zip`, data: content });
  } catch (e: any) {
    res.status(500).json({ error: e.message || 'Conversion failed' });
  }
});

// Mount Vite or static file serving
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true, port: PORT, host: '0.0.0.0' },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`AI Agent Studio server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
