Datasets are from https://zenodo.org/record/4286007#.Y5hO-C8w1-V

YAGO11K
  -entity2desc
    How:
      The first three paragraphs of Wikipedia pages without references

    Issues:
        - Manual extraction for three entities. Refer to yago_manual_descriptions.txt.
          - The first three paragraphs are taken and concatenated with a space between.
          - References are removed  
        - One of the paragraphs can be from infobox
        - Paragraphs are concatenated without space
        - 36 entity descriptions have the following as description: "Other reasons this message may be displayed:". The list of these entities is in yago_no_description.txt.
          - These are added to at the end of entity2name.txt.
        -  Labour_Party_(UK) has the following description with unknown reasons: Crown DependenciesBritish Overseas Territories
            - This is added to at the end of entity2desc.txt. Therefore, the old description is deleted from entity2desc.txt
  -entity2name
    How:
      Removing underscores from YAGO ids

WIKIDATA12K
  -entity2desc
    How:
      Jumping to Wikipedia page from corresponding Wikidata page
      The first three paragraphs of Wikipedia pages without references
      If URL is None, getting description from Wikidata page.

    Issues:
      339 entities does not have any description in both ways. See the list in wikidata_no_description.txt
      Possible reasons: the page is not english.

  -entity2name
    How:
      Title of corresponding Wikidata page
    Issues:
      34 entities do not have title e.g. https://www.wikidata.org/wiki/Q873403 (Note: Some of them have description.)
        - For manual edition please see wikidata_no_name.txt. This file is created by a) translating the label to English from another language b) Extracting from Google Knowledge graph. 1 of them is not valid ID anymore.
        - These are added to at the end of entity2name.txt and the old versions are deleted.

General issues:
  - Usually the descriptions start with the names. It causes repetition since the names and descriptions are merged.
  - "This article related to a Spanish film of the 2000s is a stub. You can help Wikipedia by expanding it." Removing this pattern.
