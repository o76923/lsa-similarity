import yaml


class SpaceSettings(object):
    dimensions: int
    remove_punctuation: bool
    remove_numbers: bool
    remove_singletons: bool
    stem: bool
    case_insensitive: bool
    stopwords: list

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return self.__dict__[item]
        else:
            return False

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def save(self, path):
        with open("%s/space_config.yml" % path, "w") as out_file:
            data = {
                "dimensions": self.dimensions,
                "stem": self.stem,
                "case_insensitive": self.case_insensitive,
                "remove": []
            }
            if self.remove_punctuation:
                data["remove"].append("punctuation")
            if self.remove_numbers:
                data["remove"].append("numbers")
            if self.remove_singletons:
                data["remove"].append("singletons")
            if self.stopwords:
                data['stopwords'] = True
                with open("%s/stopwords.txt" % path, "w") as sw_file:
                    sw_file.write("\n".join(self.stopwords))
            else:
                data['stopwords'] = False
            yaml.dump(data, out_file)
