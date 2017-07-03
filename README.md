# Latent Semantic Analysis Similarity Calculator
This container is used to perform Latent Semantic Analysis (LSA) on short 
texts. This includes the ability to create a semantic space as well as the 
ability to calculate cosine similarity scores on short texts using a semantic 
space.

## Running the Tool
In order to run the tool, type the following into your terminal

```bash
docker run -it -v /local/data/path/:/app/data -e CONFIG_FILE=config.yml o76923/lsa
```
Where `/local/data/path/` is the path on your local system that contains the 
source information and `config.yml` is the configuration file in that directory
containing which tasks should be performed as well as their settings.

## Directory Structure
The path on your local machine should contain the following
1. A configuration file
1. The semantic space
1. The texts to be compared

## Configuration File
The configuration file specifies the parameters that will tweak how the tool 
behaves. A sample configuration file is provided in `/app/conf/config.yml`.
The sections of it are as follows

### Tasks
There are two main tasks that can be performed by this tool: "create_space" and
"calculate_similarity".

#### create_space
The create_space task is used to create the semantic space that will be used
from a source paragraph given a few settings. A sample create_space task is
included below, followed by an explanation of the options available.

```yaml
  - type: create_space
    space: PR
    space_settings:
      stem: false
      case_sensitive: false
      dimensions: 500
      remove:
        - punctuation
        - singletons
        - numbers
        - stopwords:
            library: nltk
    from:
      document_scope: line
      files:
        - paragraphs/PR.txt
```

<dl>
  <dt>space</dt>
  <dd>The name that you wish to give the newly created semantic space.</dd>
  <dt>stem</dt>
  <dd>Should the words be stemmed using the Porter stemmer?</dd>
  <dt>case_sensitive</dt>
  <dd>Should words be converted to lower case before processing?</dd>
  <dt>dimensions</dt>
  <dd>What should the rank of the vectors created in the space be?</dd>
  <dt>remove</dt>
  <dd>Should punctuation, singletons, or numbers be removed? If present in
    the list, they will be removed; otherwise they are retained).</dd>
  <dt>stopwords</dt>
  <dd>If stopwords are removed, where should the list of stopword come from?
    At this time, the only option supported is to note that the stopwords list
    from nltk should be used.
  <dt>document_scope</dt>
  <dd>What defines a document for purposes of reading in source files? At the
    moment, only "line" is supported meaning each line of the file is treated
    as a different document.</dd>
  <dt>files</dt>
  <dd>A list of files that contain the source documents that you want used in
    the semantic space.</dd>
</dl>

#### calculate_similarity
The calculate_similarity task is used to generate semantic similarity scores 
between short texts. A sample calculate_similarity task is included below, 
followed by an explanation of the options available.

```yaml
  - type: calculate_similarity
    options:
      distance_metric: cosine
      space: Bus
    from:
      files:
        - input/name.txt
      pairs: all
      headers: true
      numbered: true
    output:
      format: H5
      file_name: name.h5
      ds_name: lsa_bus
```

<dl>
  <dt>space</dt>
  <dd>The name of the semantic space to be used when calculating 
    similarities.</dd>
  <dt>distance_metric</dt>
  <dd>The metric used when comparing similarities. Options are either cosine or
   r.</dd>
  <dt>files</dt>
  <dd>A list of files that contain the short texts to be compared.</dd>
  <dt>pairs</dt>
  <dd>Which pairs of short texts should be compared to one another? At the
    moment, "all" is the only option supported which compares each text to
    every other text.</dd>
  <dt>headers</dt>
  <dd>Do your files contain a header row that should be skipped?</dd>
  <dt>numbered</dt>
  <dd>Do the texts have IDs assigned to them already?</dd>
  <dt>format</dt>
  <dd>What format should the similarity scores be written to? At the moment,
    only "H5" is available which saves the output in the HDF5 file format.</dd>
  <dt>file_name</dt>
  <dd>The name of the similarity file. It will be placed in the 
    `/app/data/output` directory.</dd>
  <dt>ds_name</dt>
  <dd>If the format is "H5", you can specify the name of the data source. This
    name will be used in both the sims and vector groups.</dd>
</dl>

### Options
Options specifies global options that will apply to all tasks run. At this
time, only one option is available.
<dl>
  <dt>cores</dt>
  <dd>The number of processor cores that can be used at any given time.
</dl>

## Semantic Space
The semantic space is the corpus that is used in order to create similarity 
files. It is the output from a "create_space" task.

## Texts to be Compared
The texts to be compared are the short texts that you wish to have compared to 
one another. Similarity scores will be generated between texts with one ID and
texts with another ID.