#pragma once

#include <vector>
#include <map>
#include <functional>
#include <iostream>
#include <cassert>
#include "Utils.hpp"
#include "BakedMap.hpp"

namespace knlm
{
	using namespace std;

	class IModel
	{
	public:
		virtual size_t getVocabSize() const = 0;
		virtual size_t getOrder() const = 0;
		virtual void optimize() = 0;
		virtual void writeToStream(ostream&& str) const = 0;
		virtual void readFromStream(istream&& str) = 0;

		virtual ~IModel() {};
	};

	template<typename _WType = uint16_t>
	class KNLangModel : public IModel
	{
	public:
		using WID = _WType;
		static constexpr _WType npos = (_WType)-1;
		struct Node
		{
			friend class KNLangModel;
			class NodeIterator
			{
			protected:
				const Node * home;
				typename map<_WType, int32_t>::const_iterator mBegin;
			public:
				NodeIterator(const Node* _home, const typename map<_WType, int32_t>::const_iterator& _mBegin)
					: home(_home), mBegin(_mBegin)
				{
				}

				NodeIterator& operator++()
				{
					++mBegin;
					return *this;
				}
				bool operator==(const NodeIterator& o) const
				{
					return mBegin == o.mBegin;
				}
				bool operator!=(const NodeIterator& o) const
				{
					return !operator==(o);
				}
				pair<_WType, const Node*> operator*() const
				{
					return make_pair(mBegin->first, home + mBegin->second);
				}
			};
		protected:
			typedef function<Node*()> Allocator;
			union
			{
				map<_WType, int32_t> next;
				BakedMap<_WType, int32_t> bakedNext;
			};
		public:
			uint8_t depth = 0;
		protected:
			bool baked = false;
		public:
			int32_t parent = 0, lower = 0;
			union
			{
				uint32_t count = 0;
				float ll;
			};
			float gamma = 0;

			Node(bool _baked = false) : baked(_baked)
			{
				if (baked) new (&bakedNext) BakedMap<_WType, int32_t>();
				else new (&next) map<_WType, int32_t>();
			}

			Node(Node&& o)
			{
				if (o.baked) new (&bakedNext) BakedMap<_WType, int32_t>(move(o.bakedNext));
				else new (&next) map<_WType, int32_t>(move(o.next));

				baked = o.baked;
				swap(parent, o.parent);
				swap(lower, o.lower);
				swap(depth, o.depth);
				swap(count, o.count);
				swap(gamma, o.gamma);
			}

			~Node()
			{
				if (baked) bakedNext.~BakedMap();
				else next.~map();
			}

			Node* getParent() const
			{
				if (!parent) return nullptr;
				return (Node*)this + parent;
			}

			Node* getLower() const
			{
				if (!lower) return nullptr;
				return (Node*)this + lower;
			}

			inline Node* getNext(_WType n) const
			{
				auto it = next.find(n);
				if (it == next.end()) return nullptr;
				return (Node*)this + it->second;
			}

			inline Node* getNextFromBaked(_WType n) const
			{
				auto t = bakedNext[n];
				if (!t) return nullptr;
				return (Node*)this + t;
			}

			template<typename It>
			Node* getFromBaked(It begin, It end) const
			{
				if (begin == end) return (Node*)this;
				auto nextNode = getNextFromBaked(*begin);
				if (!nextNode) return nullptr;
				return nextNode->getFromBaked(begin + 1, end);
			}

			float getLL(_WType n, size_t endOrder) const
			{
				if (depth == endOrder)
				{
					union { int32_t t; float u; };
					t = bakedNext[n];
					if (t) return u;
				}
				else
				{
					auto* p = getNextFromBaked(n);
					if (p) return p->ll;
				}
				auto* lower = getLower();
				if (!lower) return -INFINITY;
				return gamma + lower->getLL(n, endOrder);
			}

			Node* addNextNode(_WType n, const Allocator& alloc)
			{
				Node* nextNode = alloc();
				nextNode->depth = depth + 1;
				nextNode->parent = this - nextNode;
				next[n] = nextNode - this;
				if (depth)
				{
					auto* nn = getLower()->getNext(n);
					if (!nn) nn = getLower()->addNextNode(n, alloc);
					nextNode->lower = nn - nextNode;
				}
				else nextNode->lower = nextNode->parent;
				return nextNode;
			}

			template<typename It>
			void increaseCount(It historyBegin, It historyEnd, size_t endOrder, const Allocator& alloc)
			{
				++count;
				if (historyBegin == historyEnd) return;
				if (depth == endOrder)
				{
					next[*historyBegin]++;
					return;
				}
				Node* nextNode = getNext(*historyBegin);
				if (!nextNode) nextNode = addNextNode(*historyBegin, alloc);
				nextNode->increaseCount(historyBegin + 1, historyEnd, endOrder, alloc);
			}

			void optimize()
			{
				map<_WType, int32_t> tNext = move(next);
				memset(&bakedNext, 0, sizeof(bakedNext));
				bakedNext = BakedMap<_WType, int32_t>{ tNext.begin(), tNext.end() };
				baked = true;
			}

			inline void setLL(_WType n, float ll)
			{
				next[n] = *(int32_t*)&ll;
			}

			NodeIterator begin() const
			{
				return { this, next.begin() };
			}
			NodeIterator end() const
			{
				return { this, next.end() };
			}

			void writeToStream(ostream& str, size_t leafDepth = 3) const;

			static Node readFromStream(istream& str, size_t leafDepth = 3);
		};
	protected:
		vector<Node> nodes;
		size_t orderN;
		size_t vocabSize = 0;

		void prepareCapacity(size_t minFreeSize);
		void calcDiscountedValue(size_t order, const vector<uint32_t>& cntNodes);
	public:
		KNLangModel(size_t _orderN = 3);
		KNLangModel(KNLangModel&& o)
		{
			nodes.swap(o.nodes);
			orderN = o.orderN;
			vocabSize = o.vocabSize;
		}
		size_t getVocabSize() const override { return vocabSize; }
		size_t getOrder() const override { return orderN; }
		void trainSequence(const _WType* seq, size_t len);
		void optimize() override;
		vector<float> predictNext(const _WType* history, size_t len) const;
		float evaluateLL(const _WType* seq, size_t len) const;
		float evaluateLLSent(const _WType* seq, size_t len, float minValue = -100.f) const;
		vector<float> evaluateLLEachWord(const _WType* seq, size_t len) const;
		float branchingEntropy(const _WType* seq, size_t len) const;

		void writeToStream(ostream&& str) const override
		{
			writeToBinStream<uint32_t>(str, sizeof(_WType));
			writeToBinStream<uint32_t>(str, orderN);
			writeToBinStream<uint32_t>(str, vocabSize);

			writeToBinStream<uint32_t>(str, nodes.size());
			for (auto& p : nodes)
			{
				p.writeToStream(str, orderN);
			}
		}

		KNLangModel& operator=(KNLangModel&& o)
		{
			nodes.swap(o.nodes);
			orderN = o.orderN;
			vocabSize = o.vocabSize;
			return *this;
		}

		void readFromStream(istream&& str) override
		{
			str.exceptions(istream::failbit | istream::badbit);
			nodes.clear();
			if (readFromBinStream<uint32_t>(str) > sizeof(_WType))
			{
				throw runtime_error{ "read failed. need wider size of _WType" };
			}
			orderN = readFromBinStream<uint32_t>(str);
			vocabSize = readFromBinStream<uint32_t>(str);

			uint32_t size = readFromBinStream<uint32_t>(str);
			nodes.reserve(size);
			for (size_t i = 0; i < size; ++i)
			{
				nodes.emplace_back(Node::readFromStream(str, orderN));
			}
		}

		void printStat() const;
	};

	template<typename _WType>
	KNLangModel<_WType>::KNLangModel(size_t _orderN) : orderN(_orderN)
	{
		nodes.emplace_back();
	}

	template<typename _WType>
	void KNLangModel<_WType>::prepareCapacity(size_t minFreeSize)
	{
		if (nodes.capacity() < nodes.size() + minFreeSize)
		{
			nodes.reserve(max(nodes.size() + minFreeSize, nodes.capacity() + nodes.capacity() / 2));
		}
	}

	template<typename _WType>
	void KNLangModel<_WType>::trainSequence(const _WType * seq, size_t len)
	{
		prepareCapacity(len * orderN);
		for (size_t i = 0; i < len; ++i)
		{
			nodes[0].increaseCount(seq + i, seq + min(i + orderN, len), orderN - 1, [this]()
			{
				nodes.emplace_back();
				return &nodes.back();
			});
		}
		vocabSize = max((size_t)*max_element(seq, seq + len) + 1, vocabSize);
	}

	template<typename _WType>
	void KNLangModel<_WType>::calcDiscountedValue(size_t order, const vector<uint32_t>& cntNodes)
	{
		// modified unigram probability
		if (order == 1)
		{
			size_t cntBigram = 0;
			vector<_WType> cnt;
			cnt.resize(vocabSize);
			for (auto& node : nodes)
			{
				if (node.depth != 1) continue;
				for (auto&& p : node)
				{
					cnt[p.first]++;
					cntBigram++;
				}
			}

			for (auto& p : cnt)
			{
				auto* n = nodes[0].getNext(&p - &cnt[0]);
				if (n) n->ll = p / (float)cntBigram;
			}
			return;
		}

		size_t numCount[4] = { 0, };
		map<vector<_WType>, size_t> discntNum[3];
		if (order == orderN) for (auto& node : nodes)
		{
			if (node.depth != orderN - 1) continue;
			for (auto&& p : node)
			{
				// in the leaf node
				uint32_t leafCnt = p.second - &node;
				if (leafCnt <= 4) numCount[leafCnt - 1]++;
			}
		}
		else for (auto& node : nodes)
		{
			if (node.depth != order) continue;
			auto cnt = cntNodes[&node - &nodes[0]];
			if (cnt <= 4) numCount[cnt - 1]++;
		}

		// calculating discount value
		float y = numCount[0] / (numCount[0] + 2.f * numCount[1]);
		float discntValue[3];
		for (size_t i = 0; i < 3; ++i)
		{
			discntValue[i] = numCount[i] ? (i + 1.f - (i + 2.f) * y * numCount[i + 1] / numCount[i]) : 0;
			assert(discntValue[i] >= 0);
		}

		// calculating gamma
		for (auto& node : nodes)
		{
			if (node.depth != order - 1) continue;
			size_t discntNum[3] = { 0, };
			for (auto&& p : node)
			{
				uint32_t cnt;
				// in the leaf node
				if (order == orderN) cnt = p.second - &node;
				else cnt = cntNodes[p.second - &nodes[0]];
				discntNum[min(cnt, 3u) - 1]++;
			}
			node.gamma = 0;
			for (size_t i = 0; i < 3; ++i) node.gamma += discntValue[i] * discntNum[i];
			node.gamma /= cntNodes[&node - &nodes[0]];
		}

		// applying smooth probability
		if (order == orderN) for (auto& node : nodes)
		{
			if (node.depth != orderN - 1) continue;
			for (auto&& p : node)
			{
				// in the leaf node
				uint32_t leafCnt = p.second - &node;
				float ll = (leafCnt - discntValue[min(leafCnt, 3u) - 1]) / cntNodes[&node - &nodes[0]];
				ll += node.gamma * node.getLower()->getNext(p.first)->ll;
				node.setLL(p.first, ll);
			}
		}
		else for (auto& node : nodes)
		{
			if (node.depth != order) continue;
			auto cnt = cntNodes[&node - &nodes[0]];
			node.ll = (cnt - discntValue[min(cnt, 3u) - 1]) / cntNodes[node.getParent() - &nodes[0]];
			node.ll += node.getParent()->gamma * node.getLower()->ll;
		}
	}

	template<typename _WType>
	void KNLangModel<_WType>::optimize()
	{
		{
			vector<uint32_t> cntNodes(nodes.size());
			transform(nodes.begin(), nodes.end(), cntNodes.begin(), [](const Node& n)
			{
				return n.count;
			});
			for (size_t i = 1; i <= orderN; ++i)
			{
				calcDiscountedValue(i, cntNodes);
			}
		}

		// bake likelihoods to log
		nodes[0].ll = 1;
		for (auto& node : nodes)
		{
			node.ll = log(node.ll);
			node.gamma = log(node.gamma);

			if (node.depth == orderN - 1)
			{
				for (auto&& p : node)
				{
					uint32_t t = p.second - &node;
					node.setLL(p.first, log(*(float*)&t));
				}
			}
			node.optimize();
		}
	}

	template<typename _WType>
	vector<float> KNLangModel<_WType>::predictNext(const _WType * history, size_t len) const
	{
		vector<float> prob(vocabSize);
		const Node* n = nullptr;
		for (size_t i = max(len, orderN - 1) - orderN + 1; i < len && !(n = nodes[0].getFromBaked(history + i, history + len)); ++i);
		if (!n) n = &nodes[0];
		for (size_t i = 0; i < vocabSize; ++i)
		{
			prob[i] = n->getLL(i, orderN - 1);
		}
		return prob;
	}

	template<typename _WType>
	float KNLangModel<_WType>::evaluateLL(const _WType * seq, size_t len) const
	{
		const Node* n = nullptr;
		for (size_t i = max(len - 1, orderN - 1) - orderN + 1; i < len - 1 && !(n = nodes[0].getFromBaked(seq + i, seq + len - 1)); ++i);
		if (!n) n = &nodes[0];
		return n->getLL(seq[len - 1], orderN - 1);
	}

	template<typename _WType>
	float KNLangModel<_WType>::evaluateLLSent(const _WType * seq, size_t len, float minValue) const
	{
		const KNLangModel::Node* cNode = &nodes[0];
		float score = 0;
		for (size_t i = 0; i < len; ++i)
		{
			if(i) score += max(cNode->getLL(seq[i], orderN - 1), minValue);
			if (cNode->depth == orderN - 1) cNode = cNode->getLower();
			auto nextNode = cNode->getNextFromBaked(seq[i]);
			while (!nextNode)
			{
				cNode = cNode->getLower();
				if (!cNode) break;
				nextNode = cNode->getNextFromBaked(seq[i]);
			}
			cNode = nextNode ? nextNode : &nodes[0];
		}
		return score;
	}

	template<typename _WType>
	vector<float> KNLangModel<_WType>::evaluateLLEachWord(const _WType * seq, size_t len) const
	{
		const KNLangModel::Node* cNode = &nodes[0];
		vector<float> score;
		for (size_t i = 0; i < len; ++i)
		{
			score.emplace_back(cNode->getLL(seq[i], orderN - 1));
			if (cNode->depth == orderN - 1) cNode = cNode->getLower();
			auto nextNode = cNode->getNextFromBaked(seq[i]);
			while (!nextNode)
			{
				cNode = cNode->getLower();
				if (!cNode) break;
				nextNode = cNode->getNextFromBaked(seq[i]);
			}
			cNode = nextNode ? nextNode : &nodes[0];
		}
		return score;
	}

	template<typename _WType>
	float KNLangModel<_WType>::branchingEntropy(const _WType * seq, size_t len) const
	{
		const Node* n = nullptr;
		for (size_t i = max(len, orderN - 1) - orderN + 1; i < len && !(n = nodes[0].getFromBaked(seq + i, seq + len)); ++i);
		if (!n) n = &nodes[0];
		float entropy = 0;
		for (_WType w = 0; w < vocabSize; ++w)
		{
			float p = n->getLL(w, orderN - 1);
			if (isinf(p)) continue;
			entropy -= p * exp(p);
		}
		return entropy;
	}

	template<typename _WType>
	void KNLangModel<_WType>::printStat() const
	{
		float llMin = INFINITY, llMax = -INFINITY;
		float gMin = INFINITY, gMax = -INFINITY;
		for (size_t i = 0; i < nodes.size(); ++i)
		{
			auto& n = nodes[i];
			if (isnormal(n.ll))
			{
				llMin = min(n.ll, llMin);
				llMax = max(n.ll, llMax);
			}
			if (isnormal(n.gamma))
			{
				gMin = min(n.gamma, gMin);
				gMax = max(n.gamma, gMax);
			}
		}
		cout << llMin << '\t' << llMax << endl;
		cout << gMin << '\t' << gMax << endl;
	}

	void writeNegFixed16(ostream& os, float v)
	{
		assert(v <= 0);
		auto dv = (uint16_t)min(-v * (1 << 12), 65535.f);
		writeToBinStream(os, dv);
	}

	float readNegFixed16(istream& is)
	{
		auto dv = readFromBinStream<uint16_t>(is);
		return -(dv / float(1 << 12));
	}

	template<typename _WType>
	void KNLangModel<_WType>::Node::writeToStream(ostream & str, size_t leafDepth) const
	{
		writeVToBinStream(str, -parent);
		writeSVToBinStream(str, lower);
		writeNegFixed16(str, ll);
		writeNegFixed16(str, gamma);
		writeToBinStream(str, depth);

		uint32_t size = bakedNext.size();
		writeVToBinStream(str, size);
		for (auto p : bakedNext)
		{
			writeVToBinStream(str, p.first);
			if (depth < leafDepth - 1) writeVToBinStream(str, p.second);
			else writeNegFixed16(str, *(float*)&p.second);
		}
	}

	template<typename _WType>
	typename KNLangModel<_WType>::Node KNLangModel<_WType>::Node::readFromStream(istream & str, size_t leafDepth)
	{
		Node n(true);
		n.parent = -(int32_t)readVFromBinStream(str);
		n.lower = readSVFromBinStream(str);
		n.ll = readNegFixed16(str);
		n.gamma = readNegFixed16(str);
		readFromBinStream(str, n.depth);

		uint32_t size = readVFromBinStream(str);
		vector<pair<_WType, int32_t>> tNext;
		tNext.reserve(size);
		for (size_t i = 0; i < size; ++i)
		{
			pair<_WType, int32_t> p;
			p.first = readVFromBinStream(str);
			if (n.depth < leafDepth - 1) p.second = readVFromBinStream(str);
			else
			{
				float f = readNegFixed16(str);
				p.second = *(int32_t*)&f;
			}
			tNext.emplace_back(move(p));
		}
		n.bakedNext = BakedMap<_WType, int32_t>{ tNext.begin(), tNext.end(), true };
		return n;
	}

}