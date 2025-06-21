# IS MU Chatbot: AI-Powered Assistance for Masaryk University IS

## Project Overview
This project aims to create a chatbot assistant for the Information System of Masaryk University (IS MU), using a retrieval-augmented generation (RAG) approach. The assistant draws information from official IS Help pages (Nápověda), making it easier for users to get relevant support without manually searching extensive documentation.

## Current Status and Roadmap
As of June 2025, the chatbot (ISbot) is in its initial phase. It:
- responds to single-turn questions (no conversation memory),
- primarily supports Czech language,
- is not yet publicly accessible.

Planned next steps:
- Add chat mode with multi-turn memory,
- Improve English support (currently limited due to untranslated help content),
- Expand source documents to include:
  - Study and Examination Regulations (Studijní a zkušební řád),
  - Term Calendars by Faculties (Přehled harmonogramu období fakult),
  - and possibly more.

## Model
The current prototype uses **Gemma 3 (4B)**, a multilingual open-source language model suitable for RAG-style applications.

## Motivation
The IS MU system can be complex and overwhelming, especially for new users. General-purpose language models cannot reliably assist with IS-specific tasks due to their lack of domain knowledge. This project bridges that gap by combining a language model with official IS documents, making system navigation more user-friendly.

## Project Context
This project originated within the [**PA026 Artificial Intelligence Project**](https://is.muni.cz/predmet/fi/jaro2025/PA026) course at Masaryk University.  
<!-- It was also supported financially by Masaryk University through [Projekt na podporu AI ve výuce](https://www.fi.muni.cz/~foltynek/2_2_AI_podpora_vyzva_FINAL.pdf). -->

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, contact [484757@mail.muni.cz](mailto:484757@mail.muni.cz).
