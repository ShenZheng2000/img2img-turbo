import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
from train_pix2pix_turbo import unwarp

from insightface.app import FaceAnalysis
from warp_utils.warping_layers import PlainKDEGrid, warp, invert_grid
import gradio as gr

def preprocess_image(pil_image: Image.Image) -> Image.Image:
    if pil_image is None:
        return None
    new_width = pil_image.width - pil_image.width % 8
    new_height = pil_image.height - pil_image.height % 8
    if new_width <= 0 or new_height <= 0:
        return pil_image
    return pil_image.resize((new_width, new_height), Image.LANCZOS)

def tensor_from_pil(pil_image: Image.Image) -> torch.Tensor:
    t = F.to_tensor(pil_image).unsqueeze(0).cuda()
    if args.use_fp16:
        t = t.half()
    return t

def get_saliency_bbox(pil_image: Image.Image) -> tuple:
    img_cv2 = np.array(pil_image)[:, :, ::-1]
    faces = face_app.get(img_cv2)
    height, width = pil_image.height, pil_image.width
    if len(faces) > 0:
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        x1, y1, x2, y2 = map(int, faces[0].bbox)

    else:
        x1, y1, x2, y2 = 0, 0, width, height
    return x1, y1, x2, y2

def to_pil_display(t: torch.Tensor, is_model_output: bool) -> Image.Image:
    if is_model_output:
        return transforms.ToPILImage()(t[0].cpu() * 0.5 + 0.5)
    return transforms.ToPILImage()(t[0].cpu())

def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple:
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2

 

def run_inference(input_image: Image.Image, bw: int, bbox_list=None):
    if input_image is None:
        return None, None, None
    input_image = preprocess_image(input_image)

    with torch.no_grad():
        c_t = tensor_from_pil(input_image)
        height, width = input_image.height, input_image.width

        if bw and bw > 0:
            if not isinstance(bbox_list, (list, tuple)) or len(bbox_list) != 4:
                raise ValueError('bbox_list must be a list of 4 integers [x1, y1, x2, y2]')
            x1, y1, x2, y2 = map(int, bbox_list)
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, width, height)
            bboxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32).unsqueeze(0).to(c_t.device)

            grid_net = PlainKDEGrid(
                input_shape=(height, width),
                output_shape=(height, width),
                separable=True,
                bandwidth_scale=int(bw),
                amplitude_scale=1.0,
            ).to(c_t.device)

            warp_grid = grid_net(c_t, gt_bboxes=bboxes)
            c_t_warped = warp(warp_grid, c_t)

            warped_input_pil = to_pil_display(c_t_warped, is_model_output=False)

            warped_output_tensor = model(c_t_warped, args.prompt)

            inv_grid = invert_grid(warp_grid, (1, 3, height, width), separable=True)
            output_tensor = unwarp(inv_grid, warped_output_tensor)

            warped_output_pil = to_pil_display(warped_output_tensor, is_model_output=True)
        else:
            warped_input_pil = to_pil_display(c_t, is_model_output=False)
            output_tensor = model(c_t, args.prompt)
            warped_output_pil = to_pil_display(output_tensor, is_model_output=True)

        output_pil = to_pil_display(output_tensor, is_model_output=True)
        return output_pil, warped_input_pil, warped_output_pil

def draw_bbox_on_image(pil_image: Image.Image, bbox_list):
    if pil_image is None or bbox_list is None:
        return pil_image
    x1, y1, x2, y2 = map(int, bbox_list)
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='/ssd1/shenzhen/relighting/exp_10_16/candlelight_1/test_A/39.png', help='path to the input image')
    parser.add_argument('--prompt', type=str, default='Relit with warm candlelight in a dimly lit indoor setting, casting soft, flickering shadows and enveloping the subject in golden-orange tones to create a cozy, nostalgic mood.', help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='/ssd1/shenzhen/img2img-turbo/output/pix2pix_turbo/exp_10_16_warped_512/candlelight_1/checkpoints/model_18501.pkl', help='path to a model state dict to be used')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')

    # bandwidth slider initial value
    parser.add_argument('--bw', type=int, default=512, help='bandwidth_scale for online warp')
    args = parser.parse_args()

    # Use a project-local Gradio temp/cache directory to avoid /tmp/gradio
    safe_tmp_root = os.path.join(os.getcwd(), "gradio_tmp")
    os.makedirs(safe_tmp_root, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = safe_tmp_root
    os.environ.setdefault("TMPDIR", safe_tmp_root)

    if args.model_name == '' and args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    # Load and prepare model ONCE
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()
    if args.use_fp16:
        model.half()

    # Init face detector ONCE
    face_app = FaceAnalysis(name='buffalo_l', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0)

    # Optional default input image for UI
    default_input_image = Image.open(args.input_image).convert('RGB') if os.path.exists(args.input_image) else None
    # Initialize bbox strictly via face_app when default image exists
    init_bbox_list = None
    bbox_preview_init = None
    if default_input_image is not None:
        _pre = preprocess_image(default_input_image)
        h0, w0 = _pre.height, _pre.width
        bx1, by1, bx2, by2 = get_saliency_bbox(_pre)
        bx1, by1, bx2, by2 = clamp_bbox(bx1, by1, bx2, by2, w0, h0)
        init_bbox_list = [bx1, by1, bx2, by2]
        bbox_preview_init = draw_bbox_on_image(_pre, init_bbox_list)

    with gr.Blocks(title="Relighting with Bandwidth Warp") as demo:
        gr.Markdown("**Relighting** — adjust `bandwidth_scale` to control warping strength.")
        with gr.Row():
            bw_slider = gr.Slider(minimum=0, maximum=512, value=int(args.bw), step=1, label="bandwidth_scale")
            bbox_state = gr.State(init_bbox_list)
            detect_btn = gr.Button("Auto-Detect BBox")
            run_btn = gr.Button("Run")
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Image", type="pil", value=default_input_image, width=512, height=512)
            with gr.Column():
                bbox_preview_img = gr.Image(label="Detected BBox", type="pil", value=bbox_preview_init, width=512, height=512)
            with gr.Column():
                warped_input_img = gr.Image(label="Warped Input Image", width=512, height=512)
            with gr.Column():
                warped_output_img = gr.Image(label="Warped Output Image", width=512, height=512)
            with gr.Column():
                output_img = gr.Image(label="Output Image", width=512, height=512)

        run_btn.click(fn=run_inference, inputs=[input_img, bw_slider, bbox_state], outputs=[output_img, warped_input_img, warped_output_img])

        def _auto_detect_bbox(img: Image.Image):
            if img is None:
                raise ValueError('input_image is required')
            img_proc = preprocess_image(img)
            bx1, by1, bx2, by2 = get_saliency_bbox(img_proc)
            return [bx1, by1, bx2, by2], draw_bbox_on_image(img_proc, [bx1, by1, bx2, by2])

        detect_btn.click(fn=_auto_detect_bbox, inputs=[input_img], outputs=[bbox_state, bbox_preview_img])
        input_img.change(fn=_auto_detect_bbox, inputs=[input_img], outputs=[bbox_state, bbox_preview_img])

    demo.queue()
    demo.launch(share=True)