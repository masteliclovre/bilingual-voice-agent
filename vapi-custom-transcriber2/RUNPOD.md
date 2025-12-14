# RunPod GPU deployment (quick guide)

## Goal
Run `server.py` on a RunPod GPU and point Vapi's custom transcriber URL to the RunPod public WebSocket endpoint.

## 1) Build & push a GPU image
From your machine:

```bash
docker build -t <dockerhub-user>/vapi-transcriber:gpu -f Dockerfile.gpu .
docker push <dockerhub-user>/vapi-transcriber:gpu
```

## 2) Create a RunPod GPU Pod
In RunPod:
- Create Pod -> choose a GPU (e.g. RTX 4090 / A10 / A100 depending on budget).
- Container image: `<dockerhub-user>/vapi-transcriber:gpu`
- Expose port: `8765` (TCP)
- Environment variables (recommended):
  - `MODEL_ID=lovremastelic/bva`
  - `PORT=8765`
  - (optional) `MIN_CHUNK_SECONDS=1.5`

## 3) Get the public endpoint and use it in Vapi
RunPod will give you a public host/port (or HTTP proxy URL depending on your networking option).

Your Vapi **Custom Transcriber** URL should be:

- If you have a raw TCP/WebSocket exposed:
  - `ws://<public-host>:<public-port>/api/custom-transcriber`

Important: this server accepts any path, so `/api/custom-transcriber` works even though it doesn't route by path.

## 4) Validate
From your laptop:

```bash
# Basic port check
nc -vz <public-host> <public-port>
```

Then attach it in Vapi and place a test call.

## Notes / gotchas
- If Vapi requires **wss** (TLS), use RunPod's HTTP(S) proxy / custom domain + TLS, or front it with Cloudflare/Nginx that terminates TLS and forwards to the pod.
- First boot will download the model; subsequent restarts should reuse cache if your pod uses persistent storage.
