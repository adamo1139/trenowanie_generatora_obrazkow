"""Small Gradio UI for IMG-50M (or any exported AuroraIMG2-style model), reachable from a phone on the same network.

  python app.py                                  # cpral/IMG-50M from the Hub
  python app.py --model exports/muloco50m        # a local export
  python app.py --port 7860 --gpu 0

Then open http://<this machine's LAN IP>:7860 on the phone (the address is printed at startup).
"""
import argparse
import random
import socket

import gradio as gr
import torch
from diffusers import DiffusionPipeline

EXAMPLES = [
    "a red fox sitting in a snowy forest at sunrise, photograph",
    "a futuristic city skyline at night with neon lights reflected in the rain",
    "a bowl of fresh strawberries on a wooden table, soft natural light",
    "a watercolor painting of a lighthouse on a rocky coast",
    "a small robot reading a book in a cozy library",
    "an astronaut riding a horse on the moon, digital art",
]


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cpral/IMG-50M", help="Hub repo or local export dir")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model, custom_pipeline=args.model, trust_remote_code=True).to(device)
    if device != "cpu":
        pipe.vae.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)

    def generate(prompt, steps, cfg, seed, count):
        if not prompt.strip():
            raise gr.Error("Type a prompt first")
        seed = random.randrange(2**31) if seed < 0 else int(seed)
        images = []
        for i in range(int(count)):  # one call per image so "seed + i" reproduces each one on its own
            g = torch.Generator(device=device).manual_seed(seed + i)
            images.append(pipe(prompt, num_inference_steps=int(steps), guidance_scale=float(cfg), generator=g).images[0])
        return images, f"seed {seed}" + (f" … {seed + int(count) - 1}" if count > 1 else "")

    with gr.Blocks(title="IMG-50M") as demo:
        gr.Markdown(f"### IMG-50M · 256×256 text-to-image\n`{args.model}`")
        prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a cozy cabin in a snowy forest at dusk")
        go = gr.Button("Generate", variant="primary")
        gallery = gr.Gallery(label="Images", columns=2, height="auto", object_fit="contain")
        info = gr.Markdown()
        with gr.Accordion("Settings", open=False):
            steps = gr.Slider(4, 100, value=50, step=1, label="Steps (20 = fast, 50 = best)")
            cfg = gr.Slider(1.0, 8.0, value=3.0, step=0.25, label="CFG scale")
            count = gr.Slider(1, 8, value=4, step=1, label="Images")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
        gr.Examples(EXAMPLES, inputs=prompt)
        inputs = [prompt, steps, cfg, seed, count]
        go.click(generate, inputs, [gallery, info], api_name="generate")
        prompt.submit(generate, inputs, [gallery, info])

    print(f"\nOpen on your phone: http://{lan_ip()}:{args.port}\n")
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
