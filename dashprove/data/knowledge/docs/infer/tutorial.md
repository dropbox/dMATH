* Quick Start
* Getting started with Infer
Version: 1.2.0
On this page

# Getting started with Infer

## Get Infer[​][1]

You can use our binary releases, build infer from source, or use our Docker image.

Find our latest [binary release here][2]. Download the tarball then extract it anywhere on your
system to start using infer. For example, this downloads infer in /opt on Linux (replace `VERSION`
with the latest release, eg `VERSION=1.0.0`):

`VERSION=0.XX.Y; \
curl -sSL "https://github.com/facebook/infer/releases/download/v$VERSION/infer-linux64-v$VERSION.tar
.xz" \
| sudo tar -C /opt -xJ && \
sudo ln -s "/opt/infer-linux64-v$VERSION/bin/infer" /usr/local/bin/infer
`

If the binaries do not work for you, or if you would rather build infer from source, follow the
[install from source][3] instructions to install Infer on your system.

Alternatively, use our [Docker images][4].

## Try Infer in your browser[​][5]

Try Infer on a small example on [Codeboard][6].

[1]: #get-infer
[2]: https://github.com/facebook/infer/releases/latest
[3]: https://github.com/facebook/infer/blob/main/INSTALL.md#install-infer-from-source
[4]: https://github.com/facebook/infer/tree/main/docker
[5]: #try-infer-in-your-browser
[6]: https://codeboard.io/projects/11587?view=2.1-21.0-22.0
