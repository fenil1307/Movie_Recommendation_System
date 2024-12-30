# Movie_Recommendation_System
<img width="1406" alt="Screenshot 2024-12-30 at 3 26 17 PM" src="https://github.com/user-attachments/assets/4a44123f-45e1-4f2d-a536-818c9c463877" />
<img width="1389" alt="Screenshot 2024-12-30 at 3 27 37 PM" src="https://github.com/user-attachments/assets/dac48704-e239-402c-a93e-1063966b87df" />

Overview

## Features
- **User Input**: Describe the type of movie you are looking for.
- **Filters**: Apply filters for minimum IMDb score and number of votes.
- **Dynamic Recommendations**: Suggest movies based on similarity search in the filtered dataset.
- **Interactive UI**: Built with Streamlit for a user-friendly experience.

---

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - Streamlit: Interactive UI
  - pandas: Data manipulation
  - FAISS: Fast similarity search
  - Hugging Face Transformers: Text embedding
- **Dataset**: Netflix movie titles from [this repository](https://github.com/datum-oracle/netflix-movie-titles).

---

## Installation

### Prerequisites
1. Python 3.10 or above.
2. Install required libraries using `pip`:
   ```bash
   pip install streamlit pandas sentence-transformers langchain
   ```

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
3. Open the link displayed in the terminal to access the app in your browser.


Streamlit!
[Uploading Screenshot 2024-12-30 at 3.26.17 PM.png…]()






