import os
import shutil
import uuid
from sys import argv

from pdf2image import convert_from_path
from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
import pdf_extract_kit.tasks
import re
from pytesseract import pytesseract
import cv2

def pdfToPageImageList(pdf_path,savedir):
    """
    Extracts all pages from a list of pdfs and returns a list of images.
    """
    page_image_path_list = []
    pages = convert_from_path(pdf_path, 500)
    os.makedirs(savedir, exist_ok=True)
    for i, page in enumerate(pages):
        page_image_path = f"{savedir}{i}.png"
        page.save(page_image_path, 'PNG')
        page_image_path_list.append(page_image_path)
    return page_image_path_list

def imageToImages(image_path,savedir):
    tmp_path = f"{savedir}/tmp"
    image_folder = f"{savedir}/tmp/images"
    config_path = f"{savedir}/tmp/config.yaml"
    output_folder = f"{savedir}/tmp/subimages"

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    shutil.copy(image_path, image_folder)

    config_str = f"""
    inputs:  {image_folder}
    outputs: {output_folder}
    tasks:
      layout_detection:
        model: layout_detection_yolo
        model_config:
          img_size: 1280
          conf_thres: 0.25
          iou_thres: 0.45
          batch_size: 1
          model_path: models/Layout/YOLO/yolov10l_ft.pt
          visualize: True
          rect: True
    """

    with open(config_path, "w") as f:
        f.write(config_str)

    TASK_NAME = 'layout_detection'
    config = load_config(config_path)
    task_instances = initialize_tasks_and_models(config)

    input_data = config.get('inputs', None)
    result_path = config.get('outputs', 'outputs' + '/' + TASK_NAME)

    model_layout_detection = task_instances[TASK_NAME]
    detection_results = model_layout_detection.predict_images(input_data, result_path)

    image_list = os.listdir(output_folder + '/img')

    for file in image_list:
        if 'abandon' in file:
            image_list.remove(file)

    text = ""

    paths = [output_folder + '/img/' + file for file in image_list]
    pages = [cv2.imread(output_folder + '/img/' + file) for file in image_list]

    for i,page in enumerate(pages):
        print(paths[i])
        read = pytesseract.image_to_string(page, lang="deu")
        read = read.replace('\n', ' ')
        read = read.replace('\x0c', '')
        read = re.sub(r"\s\s+", " ", read)
        print(read)

        text = text + read + '\n\n'

    shutil.rmtree(tmp_path)


    return text

def main(pdf_path):
    uu_id = uuid.uuid4()
    tmpdir = f"tmp/{uu_id}"
    os.makedirs(tmpdir, exist_ok=True)
    page_save_dir = f"{tmpdir}/pages/"
    page_image_path_list = pdfToPageImageList(pdf_path,page_save_dir)
    #print(page_image_path_list)

    page_texts = []
    for i, page_image_path in enumerate(page_image_path_list):
        page_texts.append(imageToImages(page_image_path,tmpdir))
    #print(page_texts)

    #tidy up
    shutil.rmtree(tmpdir)

    text = ""
    for page_text in page_texts:
        text = text + page_text + '\n\n'

    #save text to file
    txt_path = pdf_path.replace(".pdf",".txt")
    with open(txt_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    pdf_path = argv[1]
    main(pdf_path)
