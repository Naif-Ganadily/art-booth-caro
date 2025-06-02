# scripts/infer.py  (LoRA-free version)

from diffusers import StableDiffusionImg2ImgPipeline
import torch, argparse, pathlib, random, PIL.Image as Image

def main(inp, out):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16       # use fp16 on your 4080
    ).to("cuda")

    prompts = pathlib.Path("prompts.txt").read_text().strip().splitlines()
    prompt  = random.choice(prompts)

    init = Image.open(inp).convert("RGB").resize((512, 512))
    image = pipe(
        prompt,
        image=init,
        strength=0.65,          # 0 = keep photo, 1 = overwrite it
        guidance_scale=7.5      # “creativity slider”
    ).images[0]

    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"✨  saved → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", required=True)
    ap.add_argument("--output-path", default="output/out.png")
    args = ap.parse_args()
    main(args.input_path, args.output_path)
