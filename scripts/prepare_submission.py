import json
import argparse

# Please fill in your team information here
method = "test"  # <str> -- name of the method
team = "bytest"  # <str> -- name of the team, !!!identical to the Google Form!!!
authors = ["asdfasa"]  # <list> -- list of str, authors
email = "1832970477@qq.com"  # <str> -- e-mail address
institution = "asdff"  # <str> -- institution or company
country = "asdfasdf"  # <str> -- country or region


def main(args):
    with open(args.input, 'r') as file:
        output_res = json.load(file)

    submission_content = {
        "method": method,
        "team": team,
        "authors": authors,
        "email": email,
        "institution": institution,
        "country": country,
        "results": output_res
    }

    with open(args.output, 'w') as file:
        json.dump(submission_content, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare submission JSON.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file (the model output).")
    parser.add_argument("--output", required=True, help="Path to the input JSON file (the model output).")
    args = parser.parse_args()
    main(args)
