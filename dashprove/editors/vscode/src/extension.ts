/**
 * DashProve VS Code Extension
 *
 * Provides language support for USL (Unified Specification Language) through
 * the DashProve Language Server Protocol implementation.
 */

import * as path from 'path';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
} from 'vscode-languageclient/node';

let client: LanguageClient;

/**
 * Find the dashprove-lsp binary.
 * Priority:
 * 1. User-configured path
 * 2. Bundled binary in extension
 * 3. System PATH
 */
function findServerBinary(): string {
    const config = vscode.workspace.getConfiguration('dashprove');
    const configuredPath = config.get<string>('server.path');

    if (configuredPath && configuredPath.length > 0) {
        return configuredPath;
    }

    // Try bundled binary
    const bundledPath = path.join(__dirname, '..', 'bin', 'dashprove-lsp');
    // Fall back to system PATH
    return 'dashprove-lsp';
}

/**
 * Activate the extension.
 */
export async function activate(context: vscode.ExtensionContext): Promise<void> {
    const outputChannel = vscode.window.createOutputChannel('DashProve');
    outputChannel.appendLine('DashProve extension activating...');

    // Find server binary
    const serverPath = findServerBinary();
    outputChannel.appendLine(`Using language server: ${serverPath}`);

    const config = vscode.workspace.getConfiguration('dashprove');
    const serverArgs = config.get<string[]>('server.args') || [];

    // Server options
    const serverOptions: ServerOptions = {
        run: {
            command: serverPath,
            args: serverArgs,
            transport: TransportKind.stdio,
        },
        debug: {
            command: serverPath,
            args: serverArgs,
            transport: TransportKind.stdio,
            options: {
                env: {
                    ...process.env,
                    RUST_LOG: 'dashprove_lsp=debug',
                },
            },
        },
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'usl' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.usl'),
        },
        outputChannel,
        traceOutputChannel: outputChannel,
        initializationOptions: {
            // Pass VS Code configuration to the server
            verification: {
                autoVerify: config.get<boolean>('verification.autoVerify'),
                timeout: config.get<number>('verification.timeout'),
            },
            inlayHints: {
                enabled: config.get<boolean>('inlayHints.enabled'),
            },
            codeLens: {
                enabled: config.get<boolean>('codeLens.enabled'),
            },
        },
    };

    // Create the language client
    client = new LanguageClient(
        'dashprove',
        'DashProve Language Server',
        serverOptions,
        clientOptions
    );

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('dashprove.verify', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'usl') {
                vscode.window.showWarningMessage('Please open a USL file to verify.');
                return;
            }

            await client.sendRequest('workspace/executeCommand', {
                command: 'dashprove.verify',
                arguments: [editor.document.uri.toString()],
            });
        }),

        vscode.commands.registerCommand('dashprove.verifyProperty', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'usl') {
                vscode.window.showWarningMessage('Please open a USL file to verify.');
                return;
            }

            const position = editor.selection.active;
            await client.sendRequest('workspace/executeCommand', {
                command: 'dashprove.verifyProperty',
                arguments: [
                    editor.document.uri.toString(),
                    position.line,
                    position.character,
                ],
            });
        }),

        vscode.commands.registerCommand('dashprove.showBackendInfo', async () => {
            const info = await client.sendRequest<string>('workspace/executeCommand', {
                command: 'dashprove.showBackendInfo',
            });
            vscode.window.showInformationMessage(info);
        }),

        vscode.commands.registerCommand('dashprove.restartServer', async () => {
            outputChannel.appendLine('Restarting language server...');
            await client.restart();
            outputChannel.appendLine('Language server restarted.');
            vscode.window.showInformationMessage('DashProve language server restarted.');
        })
    );

    // Configuration change handler
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('dashprove')) {
                // Notify server of configuration change
                client.sendNotification('workspace/didChangeConfiguration', {
                    settings: vscode.workspace.getConfiguration('dashprove'),
                });
            }
        })
    );

    // Start the client
    await client.start();
    outputChannel.appendLine('DashProve extension activated.');

    // Show status bar item
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
    statusBar.text = '$(verified) DashProve';
    statusBar.tooltip = 'DashProve Language Server is running';
    statusBar.command = 'dashprove.showBackendInfo';
    statusBar.show();
    context.subscriptions.push(statusBar);
}

/**
 * Deactivate the extension.
 */
export async function deactivate(): Promise<void> {
    if (client) {
        await client.stop();
    }
}
