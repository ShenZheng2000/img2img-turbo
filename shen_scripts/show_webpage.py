import os
from natsort import natsorted

'''
cd /home/shenzhen/Relight_Projects/img2img-turbo
python3 -m http.server 8001
# use ctrl + r to refresh the page!
'''

base_dir = "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24/test_A"
result1_dir = "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/Seed_Direction_4_24/eval/fid_6501"
result2_dir = "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/Seed_Direction_4_24_use_target_bg/eval/fid_6501"
output_html = "comparison_view.html"

base_imgs = natsorted([f for f in os.listdir(base_dir) if f.endswith('.png')])
result_imgs = natsorted([f for f in os.listdir(result1_dir) if f.endswith('.png')])

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .row { display: flex; margin-bottom: 10px; }
        .row img { width: 256px; margin-right: 10px; }
    </style>
</head>
<body>
<h2>Image Comparison: Base | Result | Result (overlay relight background)</h2>
"""

# Match by index
for i in range(min(len(base_imgs), len(result_imgs))):
    base_img = base_imgs[i]
    result_img = result_imgs[i]  # Used for both result1 and result2

    print(f"Processing {base_img} and {result_img}")

    html += f"""
    <div class="row">
        <img src="{os.path.relpath(os.path.join(base_dir, base_img), start=os.path.dirname(output_html))}" alt="base">
        <img src="{os.path.relpath(os.path.join(result1_dir, result_img), start=os.path.dirname(output_html))}" alt="result1">
        <img src="{os.path.relpath(os.path.join(result2_dir, result_img), start=os.path.dirname(output_html))}" alt="result2">
    </div>
    """

html += "</body></html>"

with open(output_html, "w") as f:
    f.write(html)

print(f"[Done] HTML saved to {output_html}")