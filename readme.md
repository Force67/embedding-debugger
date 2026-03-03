## Embedding Debugger

A simple rust application that allows to generate embeddings for a set of tokens that you want to examine, can be either in json format or entered in the UI. This project does not perform inference, provide an OpenAI compatible endpoint (works with ollama too)!

Goal is to be able to quickly browse a lot of embedding points in 3D space (with PCA or t-sne) without framedrops (smooooooth), therefore this app is written in rust with iced.

### Building

"cargo run" it!
