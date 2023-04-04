
import os
import subprocess
from jiwer import wer, mer

def run_tesseract(input_image, output_text, lang):
    tesseract_cmd = ["tesseract", input_image, output_text, "-l", lang]
    subprocess.run(tesseract_cmd)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def compare_models(test_images, ground_truth_files, base_model, custom_model):
    # os.environ["TESSDATA_PREFIX"] = custom_model
    default_model_results = []
    custom_model_results = []

    for input_image, gt_file in zip(test_images, ground_truth_files):
        ground_truth_text = read_text_file(gt_file)

        # Run OCR using the default model
        # run_tesseract(input_image, "output_default.txt", "eng")
        run_tesseract(input_image, "output_default", base_model)
        default_model_text = read_text_file("output_default.txt")
        default_model_results.append((wer(ground_truth_text, default_model_text),
                                       mer(ground_truth_text, default_model_text)))

        # Run OCR using the custom model
        # run_tesseract(input_image, "output_custom.txt", "my_trained_model")
        run_tesseract(input_image, "output_custom", custom_model)
        custom_model_text = read_text_file("output_custom.txt")
        custom_model_results.append((wer(ground_truth_text, custom_model_text),
                                      mer(ground_truth_text, custom_model_text)))

    return default_model_results, custom_model_results

# test_images = ["image_1.png", "image_2.png"]  # Replace with your test image paths
# test_images = ["image_01.png", "image_02.png", "image_03.png", "image_04.png", "image_05.png", ]
test_images = ["image_01.png", ]
test_images = ["compare_model_data_results/" + x for x in test_images]

# ground_truth_files = ["image_01_gt.txt", "image_2_gt.txt"]  # Replace with your ground truth file paths
# ground_truth_files = ["image_01.gt.txt", "image_02.gt.txt", "image_03.gt.txt", "image_04.gt.txt", "image_05.gt.txt", ] 
ground_truth_files = ["image_01.gt.txt"] 
ground_truth_files = ["compare_model_data_results/" + x for x in ground_truth_files]

## name of the models we are comparing
# custom_model_dir = "/path/to/your/trained_model_dir/"
custom_model = "trained_model1"
base_model = "eng"

default_model_results, custom_model_results = compare_models(test_images, ground_truth_files,base_model, custom_model)

# Calculate average WER and CER
avg_wer_default = sum([result[0] for result in default_model_results]) / len(default_model_results)
avg_cer_default = sum([result[1] for result in default_model_results]) / len(default_model_results)
avg_wer_custom = sum([result[0] for result in custom_model_results]) / len(custom_model_results)
avg_cer_custom = sum([result[1] for result in custom_model_results]) / len(custom_model_results)

print("Default Model - Average WER:", avg_wer_default, "Average CER:", avg_cer_default)
print("Custom Model - Average WER:", avg_wer_custom, "Average CER:", avg_cer_custom)


