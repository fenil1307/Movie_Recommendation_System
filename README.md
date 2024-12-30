# Movie_Recommendation_System
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


Streamlit




