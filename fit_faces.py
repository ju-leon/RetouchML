import os
import sys
import argparse
from ffhq_dataset.face_replacement import face_replace

def main():
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('src_file', help='Image to replace faces on')
    parser.add_argument('face_file', help='File location of the face to be replaced')
    parser.add_argument('mask_file', help='File location of the mask for the face')
    parser.add_argument('face_landmarks', help='File location of the Numpy Array of the face location')
    parser.add_argument('dst_file', help='Output file location')

    args, other_args = parser.parse_known_args()

    face_replace(args.src_file, args.face_file, args.mask_file, args.face_landmarks, args.dst_file)
    print("Done!")

if __name__ == "__main__":
    main()
