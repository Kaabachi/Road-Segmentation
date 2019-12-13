from pathlib import Path
import time
from Datasets.mask_to_submission import masks_to_submission


OUTPUT_DIR = "./output"
SUBMISSION_DIR = "./submission"

if __name__ == "__main__":
    images = [str(x) for x in Path(OUTPUT_DIR).glob('**/*.png') if x.is_file()]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    submission_filename = timestr + '.csv'
    submission_path = Path(SUBMISSION_DIR) / submission_filename
    Path(SUBMISSION_DIR).mkdir(exist_ok=True)
    masks_to_submission(str(submission_path), *images)

