import numpy as np
import random

class Node:

	def __init__(self,node):
		self.id = node
		self.parent = None
		self.children = []
		self.abundance = -1.0
		self.layer = 0

	def add_child(self, child):
		self.children.append(child)

	def get_children(self):
		return self.children

	def get_children_ids(self):
		out = []
		for c in self.children:
			out.append(c.id)
		return out

	def set_parent(self, parent):
		self.parent = parent
		self.layer = parent.get_layer() + 1

	def get_parent(self):
		return self.parent

	def get_layer(self):
		return self.layer

	def get_id(self):
		return str(self.id)

	def set_id(self, id):
		self.id = id

	def get_abundance(self):
		return self.abundance

	def set_abundance(self, x):
		self.abundance = x

	def set_layer(self, x):
		self.layer = x

	def calculate_abundance(self):
		calc = 0
		for c in self.children:
			calc += c.abundance
		self.abundance = calc

	def get_leaves(self):
		calc = 0
		c = self.children
		while len(c) != 0:
			n = c.pop()
			if len(n.get_children()) == 0:
				calc = calc + 1
			else:
				for child in n.get_children():
					c.append(child)
		return calc

class Graph:
	def __init__(self):
		self.nodes = [{}]
		self.width = 0
		self.layers = 0
		self.root = None
		self.node_count = 0

	def __iter__(self):
		return iter(self.nodes.values())

	def add_node(self, l, node):
		self.node_count += 1
		self.nodes[l][node] = node

	def delete_node(self, l, node):
		self.node_count -= 1
		p = node.get_parent()
		p.get_children().remove(node)
		del(self.nodes[l][node])

	def get_node(self, l, n):
		if n in self.nodes[l]:
			return self.nodes[l][n]
		else:
			return None

	def get_node_by_name(self, n):
		for l in range(0,self.layers+1):
			for node in self.nodes[l]:
				if node.get_id() == n:
					return self.nodes[l][node]
		return None

	def get_nodes(self, l):
		return self.nodes[l]

	def get_nodes_ids(self, l):
		nlist = self.nodes[l]
		out = []
		for n in nlist:
			out.append(n.get_id())
		return sorted(out)

	def get_node_count(self):
		return self.node_count

	def get_all_nodes(self):
		list = []
		for l in range(1, self.layers+1):
			nodes = self.get_nodes(l)
			for n in nodes:
				list.append(n.get_id())
		return list
		
	def get_size(self):
		return self.layers, self.width

	def build_graph(self, mapfile='../trees/tol_species.txt'):
		self.node_count = 0
		my_map = []
		my_list=[]
		layer = -1
		node_stack = []
		current_node = None
		current_parent = None
		max_layer = 0
		num_c = 0
		skip=False
		with open(mapfile) as fd:
			for line in fd:
				segment = line.split(',')
				for sentence in segment:
					drop = sentence.count("(")
					for i in range(0,drop):
						layer = layer + 1
						node_stack.append(Node(str(layer)))
					sentence = sentence.replace("(","")
					words = sentence.split(")")
					layer = layer+1
					if (layer > max_layer):
						self.layers = layer
						max_layer = layer
						while layer >= len(self.nodes):
							self.nodes.append({})
					for w in range(0,len(words)):
						if (w == 0):
							current_node = Node(words[w].replace(".",""))
							num_c += 1
						else:
							layer = layer - 1
							current_node = node_stack.pop()
							current_node.set_id(words[w].replace(".",""))

						if (len(node_stack) > 0):
							current_parent = node_stack[-1]
							self.add_node(layer, current_node)
							current_node.set_parent(current_parent)
							current_parent.add_child(current_node)
							current_node.set_layer(layer)

						else:
							self.add_node(layer,current_node)
							current_node.set_layer(layer)
							num_c += 1
					layer = layer -1
		self.width = sum([s.get_leaves() for s in self.get_nodes(0)])

	def print_graph(self):
		for i in range(0, self.layers+1):
			for n in self.get_nodes(i):
				print(n.get_id() + " " + str(n.get_abundance()))

	def graph_vector(self):
		out = []
		for i in range(1, self.layers+1):
			for n in sorted(self.get_nodes_ids(i)):
				out.append(self.get_node_by_name(n).get_abundance())
		return out

	def graph_vector_features(self):
		out = []
		for i in range(1, self.layers+2):
			for n in sorted(self.get_nodes_ids(i)):
				name = self.get_node_by_name(n).get_id()
				out.append(self.get_node_by_name(n).get_id())
		return out
		
	def populate_graph(self, lab, x):
		# x=遍历的第一个样本。datavalue从第一行即第一个样本遍历到最后一行最后一个样本，再把第一行转置。即一个269个元素的列，代表一个样本
		# lab=f=微生物名字表
		#n1,n2计数
		n1=n2=0
		layer = self.layers
		
		level = "genus"
		if 'species' in lab.columns:
			level = 'species'
		
		for i in range(0, len(list(lab.index))):
			#lab.index为lab表中index这一列
			temp=lab.index
			abundance = x[i]
			#iloc取行号
			if lab.iloc[i][level] == "NA":
				level = 'genus'
			if lab.iloc[i][level] == "NA":
				level = 'family'
			if lab.iloc[i][level] == "NA":
				level = 'order'
			if lab.iloc[i][level] == "NA":
				level = 'class'
			if lab.iloc[i][level] == "NA":
				level = 'phylum'
			if lab.iloc[i][level] == "NA":
				level = 'kingdom'
			node = self.get_node_by_name(lab.iloc[i][level])
			if node == None:
				node = self.get_node_by_name(lab.iloc[i][level]+"_"+level)

			#-------------------------------------------------------------
			#如果在这里加权等于先加权后填树，树上会还有空的没丰度的结点不好加权
			# num_child=len(node.get_children_ids)
			# abundance=abundance*num_child
			#-------------------------------------------------------------
			node.set_abundance(abundance)
			#计数n1,n2
			n1+=1

			while node.get_parent() != None:
				p = node.get_parent()
				p.set_abundance(float(p.get_abundance()) + float(abundance))
				node = p
				n2+=1#n2计数
			level = 'species'
		# print("n1=",n1)n1=269即过滤后的微生物特征数量
		# print("n2=",n2)n2=2446??????????

	#-------------------------------------

	def routeToRoot( self ):
		layers = self.layers  # 等于实际层数,但g.get_nodes(i)的i是从1开始记录，第0层为空
		width = self.width  # 树的最大宽度
		self.write_table ( "newick_table.txt" )  # 存储所有节点与孩子的对应关系表
		# 1、按层存储每层的所有节点，即最后特征在矩阵对应的位置
		# 2、routeForAll记录所有节点(包括中间节点和叶子节点、根)到根的路径
		# 3、leaves记录所有叶子节点
		file = open ( "layer_node.txt" , "w" )
		routeForAll = {}
		leaves = [ ]
		for i in range ( 1 , layers + 1 ):  # 由于g.get_nodes(i)的i是从1开始记录，且range(a,b)是不包含b，所以g.layers+1
			i_layer_node = ""  # 用于保存第i层包含的节点
			node_list = self.get_nodes ( i )  # 第i层包含的节点列表
			for node in node_list:
				node_id = node.get_id ( )
				nodechildren = node.get_children ( )
				if nodechildren == [ ]:
					leaves.append ( node_id )
				for child in nodechildren:
					child_id = child.get_id ( )
					if node_id not in routeForAll:
						routeForAll[ child_id ] = [ node_id ]
					else:
						routeForAll[ child_id ] = [ node_id ] + routeForAll[ node_id ]
				i_layer_node = i_layer_node + " " + str ( node_id )
			file.write ( i_layer_node + "\n" )
		# print(i_layer_node)
		# print ( "leaves len: %d , leaves: %s" % (len ( leaves ) , leaves) )
		# 存储所有节点(包括中间节点和叶子节点、根)到根的路径
		routeFile = open ( "route_toRoot.txt" , "w" )
		for key , values in routeForAll.items ( ):
			# print(key, values)
			routeFile.write ( str ( key ) + ":" + " ,".join ( map ( str , values ) ) + "\n" )
		# 加入第一层第一个结点的路径即自己cellular_organisms:cellular_organisms

		routeFile.write ( "cellular_organisms:cellular_organisms" )








	def addWeight_Children( self, k ,m ) :
		#k 为系数    m为最大值
		# print ( "adding Children_num information..." )
		#----------------------------------------------------
		#查看未加权前，第一层第一个结点丰度，与加权后对比，验证是否加权了
		# nodes = self.get_nodes (1)
		# count=0
		# for node in nodes:
		# 	count+=1
		# 	print ( "unweighted:" , node.get_abundance ( ) )
		# 	print("childnum:",len(node.get_children_ids()))
		# print("count:",count)
		#-----------------------------------------------------


		for l in range(1, self.layers+1):
			nodes = self.get_nodes(l)#nodes为l层所有结点列表
			for node in nodes:
				#--------------------------------------------------
				#加权方法一：孩子结点加权
				num_child=len(node.get_children_ids())
				weight = 1 + k * num_child
				if weight>m:
					weight=m
				if weight<=1:
					weight=1
				#---------------------------------------------------



				abundance = node.get_abundance() * weight
				node.set_abundance(abundance)



		#----------------------------
		#加权后的第一层第一个结点丰度，与加权前对比验证
		# nodes = self.get_nodes(1)
		# for node in nodes:
		# 	print("weight:",node.get_abundance())
		#---------------------------------



	#-------------------------------------

	def addWeight_Height( self , k , m ,b=10):
		# print ( "adding Height information..." )
		for l in range(1, self.layers+1):
			nodes = self.get_nodes(l)#nodes为l层所有结点列表
			for node in nodes:
				height = node.get_layer()
				weight = k*height + b
				if weight>m:
					weight=m
				if weight<=1:
					weight=1


				abundance = node.get_abundance ( ) * weight
				node.set_abundance ( abundance )




	def  addWeight_Patri( self , k ,m ,mapfile='route_toRoot.txt'):
		# print("adding Patri_distance information...")
		#不管冒号前冒号后，把俩结点路径中出现的结点名数量加起来，减去重复结点即可
		#   F:C,A
		#   D:B,A
		#   Patri（F,D）=3+3-2=4
		allroute = {}
		distance = {}
		weight = {}
		sum=0
		#distance的key存储结点名，value存储该结点到其他结点的Patri距离之和
		with open ( mapfile ) as fd:
			for line in fd:
				line = line.strip ( )
				headnode = line.split ( ':' )[ 0 ]
				allroute[ headnode ] = [ headnode ] + line.split ( ":" )[ 1 ].split ( "," )
		for headnode in allroute.keys ( ):
			distance[ headnode ] = 0
			headroute = allroute[ headnode ]
			for othernode in allroute.keys ( ):
				otherroute = allroute[ othernode ]
				set1 = set ( headroute )
				set2 = set ( otherroute )
				repeatnum = len ( set1 & set2 )
				tempdistance = len ( set1 ) + len ( set2 ) - 2 * repeatnum
				distance[ headnode ] += tempdistance
				sum += distance[ headnode ]
		#必须等sum经过两层循环加完之后再求weight，所以此处for循环不可省略
		for headnode in allroute.keys():
			weight[headnode]=round(sum/((distance[headnode])*100000),2)
			weight[headnode]=k*weight[headnode]
			if weight[headnode] > m:
				weight[headnode]= m
			if weight[headnode]<=0:
				weight[headnode]=0
		print ( headnode , ":" , weight[ headnode ] )





	def get_map(self, permute=-1):
		self.set_height()
		self.set_width()
		m = np.zeros(((self.layers), self.width))
		current = self.get_nodes(1)
		if permute >= 0:
			print("Permuting Tree...")
		for i in range(0, self.layers):
			j = 0
			temp = []
			for n in current:
				m[i][j] = n.get_abundance()
				if permute >= 0:
					np.random.seed(permute)
					np.random.shuffle(n.get_children())
				temp = np.append(temp, n.get_children())
				j = j+1
			current = temp
		return m

	def get_sparse_map(self, permute=-1):
		m = np.zeros(((self.layers), self.width))
		nodes = list(self.get_nodes(1))
		j = 0
		if permute >= 0:
			print("Permuting Tree...")
		while len(nodes) > 0:
			n = nodes.pop()
			m[n.get_layer()-1][j] = n.get_abundance()
			c = n.get_children()
			if len(c) > 0:
				for child in c:
					nodes.append(child)
			else:
				j = j + 1
		return m

	def get_contrast_map(self, permute=-1):
		m = np.repeat(-1, self.layers * self.width).reshape(self.layers, self.width)
		nodes = list(self.get_nodes(1))
		j = 0
		if permute >= 0:
			print("Permuting Tree...")
		while len(nodes) > 0:
			n = nodes.pop()
			m[n.get_layer()-1][j] = n.get_abundance()
			c = n.get_children()
			if len(c) > 0:
				for child in c:
					nodes.append(child)
			else:
				j = j + 1
		return m


	def set_width(self):
		width = 1
		for i in range(0,self.layers):
			w = len(self.get_nodes(i))
			if w > width:
				width = w
		self.width = width
		
	def set_height(self):
		layer = 1
		growing = True
		drop = False
		while growing == True:
			drop = False
			for n in self.get_nodes(layer):
				if len(n.get_children()) > 0:
					drop = True
					break
			if drop == True:
				layer += 1
			else:
				growing = False
		self.layers = layer
				
	def get_ref(self):

		m = np.zeros(((self.layers), self.width), dtype=object)
		current = self.get_nodes(1)
		for i in range(0, self.layers):
			j = 0
			temp = []
			for n in current:
				m[i][j] = n.get_id()
				temp = np.append(temp, n.get_children())
				j = j+1
			current = temp
		return m
		
	def get_mask(self):

		m = np.zeros(((self.layers), self.width), dtype=object)
		current = self.get_nodes(1)
		for i in range(0, self.layers):
			j = 0
			temp = []
			for n in current:
				m[i][j] = 1.0
				temp = np.append(temp, n.get_children())
				j = j+1
			current = temp
		return m

	def write_table(self, path):
		fp = open(path, "w")
		for i in range(0, self.layers):
			node_list = self.get_nodes(i)
			for node in node_list:
				node_id = node.get_id()
				c = node.get_children()
				for child in c:
					child_id = child.get_id()
					fp.write(node_id + "\t" + child_id + "\n")

	def prune_graph(self, features_df):
		print("Pruning Tree...")
		
		if 'species' in features_df.columns:
			levels = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]
		
		else:
			levels = ["genus", "family", "order", "class", "phylum", "kingdom"]
			
		features = features_df
		for level in levels:
			try:
				features.index = features[level]
				#features['species']表示在features这个dataframe(二维表格)中'species'这一列取出来
				for f in set(list(features.index)):
					node = self.get_node_by_name(f)
					if node != None:
						features = features.drop([f])
						node.set_abundance(0)
						while node.get_parent() != None:
							p = node.get_parent()
							p.set_abundance(0)
							node = p
			except:
				continue

		total_nodes = self.get_node_count()
		i = 0
		deleted = 0
		for l in range(0, self.layers+1):
			node_list = list(self.get_nodes(l))
			for n in node_list:
				i += 1
				if n.get_abundance() == -1.0:
					self.delete_node(l, n)
					deleted += 1
			
		features = features_df
		for i in range(0, len(list(features.index))):
			found = False
			prev_node = []
			while found == False:
				for level in levels:
					if features.iloc[i][level] != "NA":
						node = self.get_node_by_name(features.iloc[i][level])

						if node == None:
							node = Node(features.iloc[i][level]+"_"+level)
							node.set_abundance(0)
							prev_node.append(node)
							
						elif node != None:
							found = True
							
							layer = node.get_layer()
							while len(prev_node) > 0:
								c = prev_node.pop()
								node.add_child(c)
								c.set_parent(node)
								self.add_node(layer+1, c)
								node = c
								layer += 1
							
					if found:
						break
		self.set_width()
		self.set_height()





	def removeRepeatName(self):
		print("removing repeat name...")
		#以下更改重复结点名称
		allnames1=[]
		namerepeat=[]
		numrepeat=0
		for l in range (self.layers+1, 0, -1):
			#倒过来从最后一层遍历到第一层
			nodes = self.get_nodes ( l )  # nodes为l层所有结点列表
			for node in nodes:
				#找到结点名字重复让从底向上改名字                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 的，然后把重复的加个字母让结点名不重复
				if node.get_id() not in allnames1:
					allnames1.append(node.get_id())
				else:
					numrepeat+=1
					oldid=node.get_id()
					namerepeat.append(oldid)
					node.set_id(oldid+'_ccnu')
		# print("numreat:",numrepeat)
		# print("namerepeat:",namerepeat)

		#------------------------------------------------------
		#结点计数
		allnames2=[]
		for l in range ( self.layers + 1 , 0 , -1 ):
			# 倒过来从最后一层遍历到第一层
			nodes = self.get_nodes ( l )  # nodes为l层所有结点列表
			for node in nodes:
				# 找到结点名字重复让从底向上改名字                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 的，然后把重复的加个字母让结点名不重复
				if node.get_id ( ) not in allnames2:
					allnames2.append ( node.get_id ( ) )
				else:

					oldid = node.get_id ( )
					namerepeat.append ( oldid )
					node.set_id ( oldid + '2' )

		countnode=0
		for l in range (self.layers+1, 0, -1):
			#倒过来从最后一层遍历到第一层
			nodes = self.get_nodes ( l )  # nodes为l层所有结点列表
			for node in nodes:
				countnode+=1
		print("countnode=",countnode)

		#---------------------------------------------------------------
		#以下算改过名字以后还有没有重复名字的结点
		print("checking repeatname...")
		namelist=[]
		for l in range(1, self.layers+1):
			nodes = self.get_nodes(l)#nodes为l层所有结点列表
			for node in nodes:#遍历，首先找到一个结点
				tempname=node.get_id()
				if tempname not in namelist:
					namelist.append(tempname)
				else:
					print("chongfu:",tempname)
		print("namenum:",len(namelist))