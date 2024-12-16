import PyPDF2
import os

def combine_pdfs(pdf_list, output_path):
    try:
        pdf_writer = PyPDF2.PdfWriter()

        for pdf in pdf_list:
            try:
                pdf_reader = PyPDF2.PdfReader(pdf)
                for page in pdf_reader.pages:
                    pdf_writer.add_page(page)
            except Exception as read_error:
                print(f"Error reading {pdf}: {read_error}")
                continue  # Skip to the next PDF if one fails

        with open(output_path, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)
    except Exception as write_error:
        print(f"Error writing to {output_path}: {write_error}")

def combine_pdfs_in_folder(folder_path, output_file):
    # List all PDF files in the folder
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    # Check if there are any PDF files to combine
    if not pdf_files:
        print("No PDF files found in the input directory.")
        return
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    combine_pdfs(pdf_files, output_file)
    print(f"Combined PDF saved as {output_file}")

# Example usage:
# combine_pdfs_in_folder(r'Files/Input_Files', 'Files/Output_File/combined.pdf')