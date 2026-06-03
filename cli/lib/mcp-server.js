const TOOLS = [
  {
    name: 'cohort_doctor',
    description: 'Check the learner\'s dev environment for required tools (Git, Python, Quarto, HF CLI, Node.js, Cursor/VS Code). Returns pass/fail for each with install hints.',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'cohort_status',
    description: 'Show the learner\'s cohort progress — which steps are complete, which are pending. Reads from local progress file.',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'cohort_done',
    description: 'Mark a cohort step as complete in the local progress file.',
    inputSchema: {
      type: 'object',
      properties: { step: { type: 'number', description: 'Step number (0-6)' } },
      required: ['step'],
    },
  },
  {
    name: 'cohort_validate',
    description: 'Run local deterministic validation checks for a cohort step. Checks files, git state, and tool presence without any LLM calls. Returns pass/fail per check.',
    inputSchema: {
      type: 'object',
      properties: { step: { type: 'number', description: 'Step number to validate (1 or 2)' } },
      required: ['step'],
    },
  },
  {
    name: 'cohort_open',
    description: 'Open a First Break AI cohort page in the default browser.',
    inputSchema: {
      type: 'object',
      properties: { page: { type: 'string', description: 'Page to open: home, roadmap, checklist, lessons, setup, blog, discord, repo, office' } },
      required: ['page'],
    },
  },
  {
    name: 'cohort_next',
    description: 'Open the next incomplete cohort step in the default browser.',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'cohort_ask',
    description: 'Ask the First Break AI assistant a question about the cohort, lessons, or AI concepts. Proxies to the remote FBA MCP server.',
    inputSchema: {
      type: 'object',
      properties: { question: { type: 'string', description: 'The question to ask' } },
      required: ['question'],
    },
  },
];

function send(msg) {
  const json = JSON.stringify(msg);
  process.stdout.write(`Content-Length: ${Buffer.byteLength(json)}\r\n\r\n${json}`);
}

function captureConsole() {
  const lines = [];
  const origLog = console.log;
  const origErr = console.error;
  console.log = (...a) => lines.push(a.map(String).join(' '));
  console.error = (...a) => lines.push(a.map(String).join(' '));
  return {
    restore: () => { console.log = origLog; console.error = origErr; },
    getText: () => lines.join('\n'),
  };
}

// Strip ANSI escape codes for clean MCP output
function stripAnsi(str) {
  return str.replace(/\x1b\[[0-9;]*m/g, '');
}

async function handleToolCall(name, args) {
  const cap = captureConsole();
  try {
    switch (name) {
      case 'cohort_doctor': {
        const mod = await import('./doctor.js');
        await mod.default();
        break;
      }
      case 'cohort_status': {
        const mod = await import('./status.js');
        await mod.default();
        break;
      }
      case 'cohort_done': {
        const mod = await import('./done.js');
        await mod.default([String(args.step)]);
        break;
      }
      case 'cohort_validate': {
        const mod = await import('./validate.js');
        await mod.default([String(args.step)]);
        break;
      }
      case 'cohort_open': {
        const mod = await import('./open.js');
        await mod.default([args.page]);
        break;
      }
      case 'cohort_next': {
        const mod = await import('./next.js');
        await mod.default();
        break;
      }
      case 'cohort_ask': {
        const mod = await import('./ask.js');
        await mod.default([args.question]);
        break;
      }
      default:
        cap.restore();
        return { error: { code: -32601, message: `Unknown tool: ${name}` } };
    }
  } catch (err) {
    cap.restore();
    return {
      content: [{ type: 'text', text: `Error: ${err.message}` }],
      isError: true,
    };
  }
  cap.restore();
  return {
    content: [{ type: 'text', text: stripAnsi(cap.getText()) }],
  };
}

async function handleMessage(msg) {
  const { id, method, params } = msg;

  switch (method) {
    case 'initialize':
      return send({
        jsonrpc: '2.0', id,
        result: {
          protocolVersion: '2024-11-05',
          capabilities: { tools: { listChanged: false } },
          serverInfo: { name: 'firstbreakai', version: '0.1.0' },
        },
      });

    case 'notifications/initialized':
      return; // no response needed

    case 'tools/list':
      return send({
        jsonrpc: '2.0', id,
        result: { tools: TOOLS },
      });

    case 'tools/call': {
      const result = await handleToolCall(params.name, params.arguments || {});
      if (result.error) {
        return send({ jsonrpc: '2.0', id, error: result.error });
      }
      return send({ jsonrpc: '2.0', id, result });
    }

    case 'ping':
      return send({ jsonrpc: '2.0', id, result: {} });

    default:
      return send({
        jsonrpc: '2.0', id,
        error: { code: -32601, message: `Method not found: ${method}` },
      });
  }
}

export default async function mcpServer() {
  let buffer = '';

  process.stdin.setEncoding('utf-8');
  process.stdin.on('data', (chunk) => {
    buffer += chunk;

    while (true) {
      const headerEnd = buffer.indexOf('\r\n\r\n');
      if (headerEnd === -1) break;

      const header = buffer.slice(0, headerEnd);
      const match = header.match(/Content-Length:\s*(\d+)/i);
      if (!match) {
        buffer = buffer.slice(headerEnd + 4);
        continue;
      }

      const contentLength = parseInt(match[1], 10);
      const bodyStart = headerEnd + 4;
      if (buffer.length < bodyStart + contentLength) break;

      const body = buffer.slice(bodyStart, bodyStart + contentLength);
      buffer = buffer.slice(bodyStart + contentLength);

      try {
        const msg = JSON.parse(body);
        handleMessage(msg).catch((err) => {
          process.stderr.write(`MCP error: ${err.message}\n`);
        });
      } catch {
        process.stderr.write('Failed to parse JSON-RPC message\n');
      }
    }
  });

  process.stdin.resume();
}
