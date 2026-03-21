'use strict';

const vscode = require('vscode');
const { execFile } = require('child_process');
const path = require('path');

/** @type {vscode.DiagnosticCollection} */
let diagnosticCollection;

/** @type {vscode.StatusBarItem} */
let statusBarItem;

/**
 * Check whether a parsed JSON document looks like an MCP schema.
 * Accepts either a bare array or an object with a "tools" array.
 * Each item must have `name` and `inputSchema` properties.
 * @param {string} text
 * @returns {boolean}
 */
function isMcpSchema(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    return false;
  }

  let tools;
  if (Array.isArray(parsed)) {
    tools = parsed;
  } else if (parsed && Array.isArray(parsed.tools)) {
    tools = parsed.tools;
  } else {
    return false;
  }

  if (tools.length === 0) return false;

  return tools.some(
    (item) =>
      item &&
      typeof item === 'object' &&
      typeof item.name === 'string' &&
      item.inputSchema !== undefined
  );
}

/**
 * Run a CLI command and return stdout as a string.
 * @param {string} cmd
 * @param {string[]} args
 * @returns {Promise<string>}
 */
function runCommand(cmd, args) {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { maxBuffer: 1024 * 1024 * 10 }, (err, stdout, stderr) => {
      if (err && err.code === 'ENOENT') {
        const notFound = new Error('ENOENT');
        notFound.code = 'ENOENT';
        reject(notFound);
        return;
      }
      // agent-friend exits non-zero when there are issues but still writes valid JSON
      // treat any output as potentially valid; only reject on ENOENT
      if (stdout && stdout.trim()) {
        resolve(stdout);
      } else if (err) {
        reject(err);
      } else {
        resolve('');
      }
    });
  });
}

/**
 * Find the line number in the document where a tool name appears.
 * Searches for `"name": "toolname"`.
 * @param {vscode.TextDocument} document
 * @param {string} toolName
 * @returns {number} zero-based line index
 */
function findToolLine(document, toolName) {
  const pattern = new RegExp(`"name"\\s*:\\s*"${escapeRegex(toolName)}"`);
  for (let i = 0; i < document.lineCount; i++) {
    if (pattern.test(document.lineAt(i).text)) {
      return i;
    }
  }
  return 0;
}

/**
 * @param {string} s
 */
function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Map a grade letter to a status bar icon codicon.
 * @param {string} grade
 * @returns {string}
 */
function gradeIcon(grade) {
  if (!grade) return '$(question)';
  const letter = grade[0].toUpperCase();
  if (letter === 'A' || letter === 'B') return '$(check)';
  if (letter === 'C') return '$(warning)';
  return '$(error)';
}

/**
 * Map an issue severity to a VS Code DiagnosticSeverity.
 * @param {string} severity
 * @returns {vscode.DiagnosticSeverity}
 */
function toDiagnosticSeverity(severity) {
  if (severity === 'error') return vscode.DiagnosticSeverity.Error;
  if (severity === 'warn' || severity === 'warning') return vscode.DiagnosticSeverity.Warning;
  return vscode.DiagnosticSeverity.Information;
}

/**
 * Show the "install agent-friend" prompt once per session.
 */
let installPromptShown = false;
async function showInstallPrompt() {
  if (installPromptShown) return;
  installPromptShown = true;
  const action = await vscode.window.showInformationMessage(
    'Install agent-friend: pip install agent-friend',
    'Copy'
  );
  if (action === 'Copy') {
    await vscode.env.clipboard.writeText('pip install agent-friend');
  }
}

/**
 * Grade and validate the given document, updating diagnostics and status bar.
 * @param {vscode.TextDocument} document
 */
async function gradeDocument(document) {
  const text = document.getText();

  if (!isMcpSchema(text)) {
    // Not an MCP schema — stay silent, clear any previous state for this file
    diagnosticCollection.delete(document.uri);
    statusBarItem.hide();
    return;
  }

  const filePath = document.uri.fsPath;

  // Run validate and grade in parallel
  let validateResult, gradeResult;
  try {
    [validateResult, gradeResult] = await Promise.all([
      runCommand('agent-friend', ['validate', '--json', filePath]),
      runCommand('agent-friend', ['grade', '--json', filePath]),
    ]);
  } catch (err) {
    if (err && err.code === 'ENOENT') {
      await showInstallPrompt();
      return;
    }
    // Other errors: surface briefly to output, stay silent in UI
    console.error('[agent-friend] command error:', err);
    return;
  }

  // --- Parse validate output and push diagnostics ---
  const diagnostics = [];
  try {
    const v = JSON.parse(validateResult);
    if (Array.isArray(v.issues)) {
      for (const issue of v.issues) {
        const line = issue.tool ? findToolLine(document, issue.tool) : 0;
        const lineText = document.lineAt(line).text;
        const range = new vscode.Range(line, 0, line, lineText.length);
        const diag = new vscode.Diagnostic(
          range,
          `[${issue.check}] ${issue.message}`,
          toDiagnosticSeverity(issue.severity)
        );
        diag.source = 'agent-friend';
        diagnostics.push(diag);
      }
    }
  } catch {
    // Validate output not parseable — skip diagnostics
  }
  diagnosticCollection.set(document.uri, diagnostics);

  // --- Parse grade output and update status bar ---
  try {
    const g = JSON.parse(gradeResult);
    const grade = g.overall_grade || '?';
    const score = g.overall_score != null ? Math.round(g.overall_score) : '?';
    const icon = gradeIcon(grade);
    statusBarItem.text = `${icon} MCP: ${grade} (${score}/100)`;
    statusBarItem.tooltip = [
      `Correctness: ${g.correctness ? g.correctness.grade : '?'}`,
      `Efficiency:  ${g.efficiency ? g.efficiency.grade : '?'}`,
      `Quality:     ${g.quality ? g.quality.grade : '?'}`,
      `Tools: ${g.tool_count || '?'} | Tokens: ${g.total_tokens || '?'}`,
    ].join('\n');
    statusBarItem.command = 'agent-friend.grade';
    statusBarItem.show();
  } catch {
    statusBarItem.hide();
  }
}

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  diagnosticCollection = vscode.languages.createDiagnosticCollection('agent-friend');
  context.subscriptions.push(diagnosticCollection);

  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  context.subscriptions.push(statusBarItem);

  // Grade active editor on activation
  if (vscode.window.activeTextEditor) {
    gradeDocument(vscode.window.activeTextEditor.document);
  }

  // Grade on open
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((doc) => {
      if (doc.languageId === 'json') {
        gradeDocument(doc);
      }
    })
  );

  // Grade on save
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (doc.languageId === 'json') {
        gradeDocument(doc);
      }
    })
  );

  // Hide status bar when switching away from a JSON file
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      if (!editor || editor.document.languageId !== 'json') {
        statusBarItem.hide();
      } else {
        gradeDocument(editor.document);
      }
    })
  );

  // Manual command
  context.subscriptions.push(
    vscode.commands.registerCommand('agent-friend.grade', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active editor.');
        return;
      }
      gradeDocument(editor.document);
    })
  );
}

function deactivate() {
  if (diagnosticCollection) {
    diagnosticCollection.clear();
    diagnosticCollection.dispose();
  }
  if (statusBarItem) {
    statusBarItem.hide();
    statusBarItem.dispose();
  }
}

module.exports = { activate, deactivate };
