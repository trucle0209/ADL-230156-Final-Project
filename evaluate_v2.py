"""Lightweight 20-question evaluation for Applied Mathematics RAG V2.

Run one case at a time; append each result to evaluation_results/results_v2.csv.
Question answers are reference targets for manual verification, not automatic grading.
"""
import argparse
import csv
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from run_pdf import EMBED_MODEL, OLLAMA_URL, load_and_index, ask

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / 'evaluation_results' / 'results_v2.csv'

# (category, question, reference answer, reference PDF page)
QUESTIONS = {
    'Q01': ('Major requirements', 'How many credits are required for the Applied Mathematics major?', '56 credits', '9'),
    'Q02': ('Graduation', 'What is the minimum total number of credits required for graduation?', '128 credits', '10'),
    'Q03': ('Declaration', 'How many MATH courses must students complete before declaring the major?', 'At least 2 MATH courses', '10'),
    'Q04': ('Capstone', 'If an Applied Mathematics student does not take Capstone I and II, what must they complete instead?', '2 advanced courses at the 300-level', '9'),
    'Q05': ('Minor', 'How many credits are required for the Applied Mathematics minor?', '24 credits', '10'),
    'Q06': ('Minor', 'What is the minimum number of MATH courses needed for the Applied Mathematics minor?', 'At least 6 MATH courses', '10'),
    'Q07': ('Minor', 'How many minor MATH courses must be at the 200-level or above?', 'At least 3 courses', '10'),
    'Q08': ('Minor', 'Among the courses for the minor, how many must be at the 300-level?', 'At least 1 course', '10'),
    'Q09': ('Course structure', 'How many intermediate mathematics courses must Applied Mathematics majors take?', 'At least 2 intermediate courses', '10'),
    'Q10': ('Tracks', 'What are the three specialized tracks in the Applied Mathematics program?', 'Data Science; Engineering Mathematics; Quantitative Finance', '7'),
    'Q11': ('Tracks', 'Which track focuses on statistical modeling, machine learning, and big data analytics?', 'Data Science', '7'),
    'Q12': ('Tracks', 'Which track focuses on computational mathematics and numerical methods used in engineering?', 'Engineering Mathematics', '7'),
    'Q13': ('Tracks', 'Which track focuses on financial modeling, risk management, and asset pricing?', 'Quantitative Finance', '7'),
    'Q14': ('Applied electives', 'How many elective applied courses must Applied Mathematics majors take?', 'At least 2 elective applied courses', '12'),
    'Q15': ('Applied electives', 'What is the minimum level requirement for elective applied courses?', 'At least 1 applied course at the 300-level', '12'),
    'Q16': ('Academic policy', 'Can students count the same course toward two different major requirements?', 'No, double-counting is not allowed', '13'),
    'Q17': ('Academic policy', 'Can Optimization count simultaneously as an Advanced course and an Elective Applied course?', 'No. It can count toward either one, but not both', '13'),
    'Q18': ('Sample study plan', 'In sample Model A, which academic year is reserved for Capstone I and Capstone II?', 'Year 4 (senior year)', '15'),
    'Q19': ('Core curriculum', 'How many core courses and credits are listed under General Curriculum Requirements?', '4 core courses, 16 credits', '9'),
    'Q20': ('Out of scope', 'What is the exact tuition fee for the Applied Mathematics major in the year 2030?', 'Not stated in the handbook; should decline to invent a number', 'N/A'),
}

FIELDNAMES = [
    'question_id', 'version', 'retrieval', 'model', 'top_k',
    'question', 'expected_answer', 'actual_answer', 'source_pages',
    'response_time_seconds', 'manual_score', 'notes',
]


def completed_keys():
    if not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0:
        return set()
    with OUTPUT_FILE.open('r', encoding='utf-8-sig', newline='') as file:
        return {
            (r.get('question_id', ''), r.get('model', ''),
             r.get('top_k', ''), r.get('retrieval', ''))
            for r in csv.DictReader(file)
        }


def save_row(row):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0
    with OUTPUT_FILE.open('a', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if needs_header:
            writer.writeheader()
        # Fieldnames remain identical to the previous 4-question evaluate_v2.py.
        writer.writerow({field: row.get(field, '') for field in FIELDNAMES})
        file.flush()


def main():
    parser = argparse.ArgumentParser(description='MATH RAG V2 | 20-question evaluation')
    parser.add_argument('--question-id', choices=[*QUESTIONS, 'all'], default='Q05')
    parser.add_argument('--list', action='store_true', help='List 20 evaluation questions; no models loaded')
    parser.add_argument('--model', default='qwen2.5:1.5b')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retrieval', choices=['semantic', 'hybrid'], default='hybrid')
    args = parser.parse_args()

    if args.list:
        for qid, (category, question, expected, page) in QUESTIONS.items():
            print(f'{qid} | {category} | {question}')
            print(f'     Reference: {expected} | PDF page: {page}')
        return

    if args.top_k < 1:
        parser.error('--top-k must be >= 1')

    chosen = list(QUESTIONS) if args.question_id == 'all' else [args.question_id]
    completed = completed_keys()
    pending = [qid for qid in chosen if (qid, args.model, str(args.top_k), args.retrieval) not in completed]

    print('\n' + '=' * 52)
    print('           MATH RAG V2 | EVALUATION')
    print('=' * 52)
    print(f'Model: {args.model} | Retrieval: {args.retrieval} | Top-K: {args.top_k}')
    print(f'Pending questions: {len(pending)}')
    print('-' * 52)
    if not pending:
        print(f'[SKIP] Already saved to {OUTPUT_FILE}')
        return

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    db = load_and_index(embeddings)
    llm = OllamaLLM(model=args.model, base_url=OLLAMA_URL, temperature=0)

    for qid in pending:
        category, question, expected, reference_page = QUESTIONS[qid]
        print(f'\n[{qid}] {question}')
        try:
            answer, sources, elapsed = ask(
                question=question, db=db, llm=llm,
                top_k=args.top_k, retrieval=args.retrieval,
            )
            pages = sorted({str(doc.metadata.get('page', '?')) for doc, _ in sources}, key=lambda x: int(x) if x.isdigit() else 9999)
            save_row({
                'question_id': qid,
                'version': 'V2',
                'retrieval': args.retrieval,
                'model': args.model,
                'top_k': args.top_k,
                'question': question,
                'expected_answer': expected,
                'actual_answer': answer,
                'source_pages': ', '.join(pages),
                'response_time_seconds': round(elapsed, 2),
                'manual_score': '',
                'notes': '',
            })
            print(f'Expected: {expected} (reference page {reference_page})')
            print(f'Actual  : {answer}')
            print(f'Pages   : {", ".join(pages)}')
            print(f'Time    : {elapsed:.2f}s')
            print('[SAVED] Manual score intentionally left blank.')
        except KeyboardInterrupt:
            print('\n[STOPPED] Earlier rows were saved.')
            break
        except Exception as exc:
            print(f'[ERROR] {type(exc).__name__}: {exc}')
            break

    print(f'\nCSV: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()