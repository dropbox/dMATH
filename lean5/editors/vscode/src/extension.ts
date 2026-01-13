import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('lean5');
    const serverPath = config.get<string>('serverPath', 'lean5-lsp');

    const serverOptions: ServerOptions = {
        run: {
            command: serverPath,
            transport: TransportKind.stdio
        },
        debug: {
            command: serverPath,
            transport: TransportKind.stdio
        }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'lean5' },
            { scheme: 'file', language: 'lean' }
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.lean')
        }
    };

    client = new LanguageClient(
        'lean5',
        'Lean5 Language Server',
        serverOptions,
        clientOptions
    );

    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
