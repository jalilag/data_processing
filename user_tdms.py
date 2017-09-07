from nptdms import TdmsFile

class UTdms(TdmsFile):
	def __init__(self,file, memmap_dir=None):
		super().__init__(file, memmap_dir=None)

	def get_params(self):
		return self.object().properties

	def get_param(self,key_name):
		return self.object().property(key_name)

	def get_groups(self):
		return self.groups()

	def get_channels(self,group_name):
		g = self.get_groups()
		if group_name in g:
			return self.group_channels(group_name)
		else:
			if isinstance(group_name,int):
				if len(g) > group_name:
					return self.group_channels(g[group_name])
			else:
				if group_name.isdigit(): 
					if len(g) > int(group_name):
						return self.group_channels(g[int(group_name)])
		return list()

	def get_channel(self,group_name,channel_name):
		g = self.get_channels(group_name)
		if len(g) > 0:
			for i in g: 
				if channel_name == i.channel:
					return i
			if isinstance(channel_name,int):
				if len(g) > channel_name:
					return g[channel_name]
			if channel_name.isdigit():
				if len(g) > int(channel_name):
					return g[int(channel_name)]
		return list()