/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout.
 * Uses the GitHub Copilot SDK (@github/copilot-sdk) with Claude models via BYOK
 * (Anthropic API key routed through the host credential proxy) or native GitHub
 * Copilot authentication (GH_TOKEN).
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json — polled and consumed
 *          Sentinel: /workspace/ipc/input/_close — signals session end
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per assistant message).
 *   Final marker after loop ends signals completion.
 */

import fs from 'fs';
import path from 'path';
import { CopilotClient, approveAll } from '@github/copilot-sdk';
import type { CopilotSession } from '@github/copilot-sdk';
import { fileURLToPath } from 'url';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

interface MessageEntry {
  role: 'user' | 'assistant';
  content: string;
}

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

/**
 * Check for _close sentinel.
 */
function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }
    return true;
  }
  return false;
}

/**
 * Drain all pending IPC input messages.
 */
function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs.readdirSync(IPC_INPUT_DIR)
      .filter(f => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(`Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`);
        try { fs.unlinkSync(filePath); } catch { /* ignore */ }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

/**
 * Wait for a new IPC message or _close sentinel.
 * Returns the messages joined as a single string, or null if _close.
 */
function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) {
        resolve(null);
        return;
      }
      const messages = drainIpcInput();
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

/**
 * Archive the conversation to conversations/ at session end.
 */
async function archiveConversation(
  session: CopilotSession,
  assistantName?: string,
): Promise<void> {
  try {
    const events = await session.getMessages();
    const messages: MessageEntry[] = [];
    for (const event of events) {
      if (event.type === 'user.message') {
        const e = event as { data?: { content?: string } };
        if (e.data?.content) messages.push({ role: 'user', content: e.data.content });
      } else if (event.type === 'assistant.message') {
        const e = event as { data?: { content?: string } };
        if (e.data?.content) messages.push({ role: 'assistant', content: e.data.content });
      }
    }
    if (messages.length === 0) return;

    const conversationsDir = '/workspace/group/conversations';
    fs.mkdirSync(conversationsDir, { recursive: true });

    const now = new Date();
    const date = now.toISOString().split('T')[0];
    const time = `${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}`;
    const filename = `${date}-conversation-${time}.md`;
    const filePath = path.join(conversationsDir, filename);

    const lines: string[] = ['# Conversation', '', `Archived: ${now.toLocaleString()}`, '', '---', ''];
    for (const msg of messages) {
      const sender = msg.role === 'user' ? 'User' : (assistantName || 'Assistant');
      const content = msg.content.length > 2000 ? msg.content.slice(0, 2000) + '...' : msg.content;
      lines.push(`**${sender}**: ${content}`, '');
    }
    fs.writeFileSync(filePath, lines.join('\n'));
    log(`Archived conversation to ${filePath}`);
  } catch (err) {
    log(`Failed to archive conversation: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/**
 * Run a single query: send a prompt and wait for the session to become idle.
 * Concurrently polls for IPC messages and pipes them into the session.
 *
 * Returns whether a _close sentinel was encountered during the query.
 */
function runQuery(
  session: CopilotSession,
  prompt: string,
  onIdle: () => void,
): Promise<{ closedDuringQuery: boolean }> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let closedDuringQuery = false;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;

    const done = (closed: boolean) => {
      if (settled) return;
      settled = true;
      if (pollTimer) clearTimeout(pollTimer);
      resolve({ closedDuringQuery: closed });
    };

    // session.idle fires when the session has finished processing all queued messages
    const unsubIdle = session.on('session.idle', () => {
      unsubIdle();
      onIdle();
      done(closedDuringQuery);
    });

    // IPC polling — runs while the session is processing
    const pollIpc = () => {
      if (settled) return;
      if (shouldClose()) {
        closedDuringQuery = true;
        // Don't force-close yet; wait for session.idle to fire naturally
        // so the current agent turn can complete.
        return;
      }
      const messages = drainIpcInput();
      for (const msg of messages) {
        log(`Piping IPC message into active session (${msg.length} chars)`);
        session.send({ prompt: msg }).catch((err: unknown) => {
          log(`Failed to queue IPC message (message will be dropped): ${err instanceof Error ? err.message : String(err)}`);
        });
      }
      pollTimer = setTimeout(pollIpc, IPC_POLL_MS);
    };
    pollTimer = setTimeout(pollIpc, IPC_POLL_MS);

    // Send the prompt (async, queues the message)
    session.send({ prompt }).catch((err: unknown) => {
      if (settled) return;
      settled = true;
      if (pollTimer) clearTimeout(pollTimer);
      unsubIdle();
      reject(err);
    });
  });
}

async function main(): Promise<void> {
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData);
    try { fs.unlinkSync('/tmp/input.json'); } catch { /* may not exist */ }
    log(`Received input for group: ${containerInput.groupFolder}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`
    });
    process.exit(1);
  }

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel from previous container runs
  try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }

  // Auth: BYOK Anthropic (via credential proxy) or GitHub Copilot native (GITHUB_COPILOT_TOKEN)
  //
  // BYOK Anthropic:   NANOCLAW_PROXY_URL is set; no GITHUB_COPILOT_TOKEN.
  //   Container's Copilot SDK uses provider.baseUrl pointing to the credential proxy.
  //   The proxy injects the real ANTHROPIC_API_KEY; the container never sees it.
  //
  // GitHub Copilot native:  GITHUB_COPILOT_TOKEN is set; no NANOCLAW_PROXY_URL.
  //   CopilotClient uses the token directly; the Copilot CLI authenticates with GitHub.
  const proxyUrl = process.env.NANOCLAW_PROXY_URL;
  const githubToken = process.env.GITHUB_COPILOT_TOKEN;
  const model = process.env.NANOCLAW_MODEL || 'claude-sonnet-4.5';

  // Load global CLAUDE.md as additional system context (shared across all groups)
  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  let globalClaudeMd: string | undefined;
  if (!containerInput.isMain && fs.existsSync(globalClaudeMdPath)) {
    globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
  }

  // Create the Copilot client
  const client = new CopilotClient({
    githubToken: githubToken || undefined,
  });

  try {
    await client.start();
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to start Copilot client: ${err instanceof Error ? err.message : String(err)}`
    });
    process.exit(1);
  }

  // Provider config for BYOK Anthropic mode (routes through credential proxy)
  const provider = proxyUrl ? {
    type: 'anthropic' as const,
    baseUrl: proxyUrl,
    apiKey: process.env.ANTHROPIC_API_KEY || 'placeholder',
  } : undefined;

  // Skills directory synced from host container/skills/
  const skillsDir = '/home/node/.copilot/skills';
  const skillDirectories = fs.existsSync(skillsDir) ? [skillsDir] : undefined;

  // Build initial prompt (drain any pending IPC messages too)
  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  // Session configuration
  const sessionConfig = {
    model,
    provider,
    workingDirectory: '/workspace/group',
    configDir: '/home/node/.copilot',
    systemMessage: globalClaudeMd ? { content: globalClaudeMd } : undefined,
    skillDirectories,
    mcpServers: {
      nanoclaw: {
        type: 'local' as const,
        command: 'node',
        args: [mcpServerPath],
        tools: ['*'] as string[],
        env: {
          NANOCLAW_CHAT_JID: containerInput.chatJid,
          NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
          NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
        },
      },
    },
    onPermissionRequest: approveAll,
    infiniteSessions: { enabled: true },
  };

  // Create or resume session
  let session: CopilotSession;
  const sessionId = containerInput.sessionId;

  try {
    if (sessionId) {
      log(`Resuming session: ${sessionId}`);
      try {
        session = await client.resumeSession(sessionId, sessionConfig);
        log(`Session resumed: ${session.sessionId}`);
      } catch (resumeErr) {
        log(`Resume failed (${resumeErr instanceof Error ? resumeErr.message : String(resumeErr)}), creating new session`);
        session = await client.createSession(sessionConfig);
        log(`New session created: ${session.sessionId}`);
      }
    } else {
      session = await client.createSession(sessionConfig);
      log(`Session created: ${session.sessionId}`);
    }
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to create session: ${err instanceof Error ? err.message : String(err)}`
    });
    await client.stop();
    process.exit(1);
  }

  const newSessionId = session.sessionId;

  // Register assistant.message handler — fires for each complete assistant response
  session.on('assistant.message', (event) => {
    const raw = event.data.content;
    // Strip <internal>...</internal> blocks — the agent wraps internal reasoning
    // or tool-call commentary in these tags to keep them out of user-visible output.
    const text = raw.replace(/<internal>[\s\S]*?<\/internal>/g, '').trim();
    log(`Assistant message: ${raw.slice(0, 200)}`);
    if (text) {
      writeOutput({ status: 'success', result: text, newSessionId });
    }
  });

  // Main query loop
  try {
    while (true) {
      log(`Starting query (session: ${newSessionId}, prompt: ${prompt.length} chars)`);

      const { closedDuringQuery } = await runQuery(
        session,
        prompt,
        () => log(`Session idle after query`),
      );

      // Emit session-update marker so host can track the session ID
      writeOutput({ status: 'success', result: null, newSessionId });

      if (closedDuringQuery || shouldClose()) {
        log('Close sentinel consumed during query, exiting');
        break;
      }

      log('Query complete, waiting for next IPC message...');

      // Wait for the next message or _close sentinel
      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId,
      error: errorMessage
    });
    await session.disconnect();
    await client.stop();
    process.exit(1);
  }

  // Archive conversation before exiting
  await archiveConversation(session, containerInput.assistantName);

  await session.disconnect();
  await client.stop();
}

main();
