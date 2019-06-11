#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>
#include <Python.h>

#include "KNLangModel.hpp"

using namespace std;

static PyObject *gModule, *gClass;
static PyObject* knlm__init(PyObject* self, PyObject* args)
{
	PyObject* argSelf;
	size_t numOrder = 3, wordSize = 2;
	if (!PyArg_ParseTuple(args, "O|nn", &argSelf, &numOrder, &wordSize)) return nullptr;
	try
	{
		ssize_t inst = 0;
		switch (wordSize)
		{
		case 1:
			inst = (ssize_t)new knlm::KNLangModel<uint8_t>{ numOrder };
			break;
		case 2:
			inst = (ssize_t)new knlm::KNLangModel<uint16_t>{ numOrder };
			break;
		case 4:
			inst = (ssize_t)new knlm::KNLangModel<uint32_t>{ numOrder };
			break;
		default:
			throw runtime_error{ "wordSize must be 1, 2 or 4" };
		}
		PyObject_SetAttrString(argSelf, "_inst", PyLong_FromLongLong(inst));
		PyObject_SetAttrString(argSelf, "_wsize", Py_BuildValue("n", wordSize));
		PyObject* dict = PyDict_New();
		PyDict_SetItemString(dict, "___UNK___", Py_BuildValue("n", 0));
		PyDict_SetItemString(dict, "___BEG___", Py_BuildValue("n", 1));
		PyDict_SetItemString(dict, "___END___", Py_BuildValue("n", 2));
		PyObject_SetAttrString(argSelf, "_dict", dict);
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* knlm__del(PyObject* self, PyObject* args)
{
	PyObject* argSelf;
	if (!PyArg_ParseTuple(args, "O", &argSelf)) return nullptr;
	try
	{
		PyObject* instObj = PyObject_GetAttrString(argSelf, "_inst");
		if (!instObj) throw runtime_error{ "_inst is null" };
		PyObject* wsizeObj = PyObject_GetAttrString(argSelf, "_wsize");
		knlm::IModel* inst = (knlm::IModel*)PyLong_AsLongLong(instObj);
		size_t wsize = PyLong_AsLong(wsizeObj);
		Py_DECREF(instObj);
		Py_DECREF(wsizeObj);
		if (inst) delete inst;
		PyObject_SetAttrString(argSelf, "_inst", PyLong_FromLongLong(0));
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}

	Py_INCREF(Py_None);
	return Py_None;
}

template<typename _WType>
vector<_WType> makeSeqList(PyObject *iter, PyObject* dict)
{
	PyObject* item;
	vector<_WType> seq;
	seq.emplace_back(1);
	while (item = PyIter_Next(iter))
	{
		PyObject* idx = PyDict_GetItem(dict, item);
		size_t id = 0;
		if (idx)
		{
			id = PyLong_AsLong(idx);
		}
		else
		{
			PyDict_SetItem(dict, item, Py_BuildValue("n", id = PyDict_Size(dict)));
			if (sizeof(_WType) < 4 && id >= (1 << (sizeof(_WType) * 8)))
			{
				throw runtime_error{ "" };
			}
		}
		seq.emplace_back(id);
		Py_DECREF(item);
	}
	seq.emplace_back(2);
	return seq;
}

template<typename _WType>
vector<_WType> makeSeqListConst(PyObject *iter, PyObject* dict, bool end = true)
{
	PyObject* item;
	vector<_WType> seq;
	seq.emplace_back(1);
	while (item = PyIter_Next(iter))
	{
		PyObject* idx = PyDict_GetItem(dict, item);
		size_t id = 0;
		if (idx) id = PyLong_AsLong(idx);
		seq.emplace_back(id);
		Py_DECREF(item);
	}
	if (end) seq.emplace_back(2);
	return seq;
}


static PyObject* knlm__train(PyObject* self, PyObject* args)
{
	PyObject *argSelf, *argIter, *item;
	if (!PyArg_ParseTuple(args, "OO", &argSelf, &argIter)) return nullptr;
	try
	{
		PyObject* instObj = PyObject_GetAttrString(argSelf, "_inst");
		if (!instObj) throw runtime_error{ "_inst is null" };
		PyObject* wsizeObj = PyObject_GetAttrString(argSelf, "_wsize");
		knlm::IModel* inst = (knlm::IModel*)PyLong_AsLongLong(instObj);
		size_t wsize = PyLong_AsLong(wsizeObj);
		Py_DECREF(instObj);
		Py_DECREF(wsizeObj);
		if (!(argIter = PyObject_GetIter(argIter)))
		{
			throw runtime_error{ "argIter is not iterable" };
		}

		PyObject* dict = PyObject_GetAttrString(argSelf, "_dict");
		try
		{
			if (wsize == 1)
			{
				auto seq = makeSeqList<uint8_t>(argIter, dict);
				((knlm::KNLangModel<uint8_t>*)inst)->trainSequence(&seq[0], seq.size());
			}
			else if (wsize == 2)
			{
				auto seq = makeSeqList<uint16_t>(argIter, dict);
				((knlm::KNLangModel<uint16_t>*)inst)->trainSequence(&seq[0], seq.size());
			}
			else if (wsize == 4)
			{
				auto seq = makeSeqList<uint32_t>(argIter, dict);
				((knlm::KNLangModel<uint32_t>*)inst)->trainSequence(&seq[0], seq.size());
			}
		}
		catch (const runtime_error&)
		{
			Py_DECREF(dict);
			Py_DECREF(argIter);
			PyErr_Format(PyExc_RuntimeError, "vocab size overflow. use bigger 'wsize' than %d", wsize);
			return nullptr;
		}
		Py_DECREF(dict);
		Py_DECREF(argIter);
		Py_INCREF(Py_None);
		return Py_None;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* knlm__optimize(PyObject* self, PyObject* args)
{
	PyObject *argSelf;
	if (!PyArg_ParseTuple(args, "O", &argSelf)) return nullptr;
	try
	{
		PyObject* instObj = PyObject_GetAttrString(argSelf, "_inst");
		if (!instObj) throw runtime_error{ "_inst is null" };
		PyObject* wsizeObj = PyObject_GetAttrString(argSelf, "_wsize");
		knlm::IModel* inst = (knlm::IModel*)PyLong_AsLongLong(instObj);
		size_t wsize = PyLong_AsLong(wsizeObj);
		Py_DECREF(instObj);
		Py_DECREF(wsizeObj);
		inst->optimize();
		Py_INCREF(Py_None);
		return Py_None;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* knlm__evaluate(PyObject* self, PyObject* args)
{
	PyObject *argSelf, *argIter, *item;
	if (!PyArg_ParseTuple(args, "OO", &argSelf, &argIter)) return nullptr;
	try
	{
		PyObject* instObj = PyObject_GetAttrString(argSelf, "_inst");
		if (!instObj) throw runtime_error{ "_inst is null" };
		PyObject* wsizeObj = PyObject_GetAttrString(argSelf, "_wsize");
		knlm::IModel* inst = (knlm::IModel*)PyLong_AsLongLong(instObj);
		size_t wsize = PyLong_AsLong(wsizeObj);
		Py_DECREF(instObj);
		Py_DECREF(wsizeObj);

		if (!(argIter = PyObject_GetIter(argIter)))
		{
			throw runtime_error{ "argIter is not iterable" };
		}

		PyObject* dict = PyObject_GetAttrString(argSelf, "_dict");
		float score = 0;
		if (wsize == 1)
		{
			auto seq = makeSeqListConst<uint8_t>(argIter, dict, false);
			score = ((knlm::KNLangModel<uint8_t>*)inst)->evaluateLL(&seq[0], seq.size());
		}
		else if (wsize == 2)
		{
			auto seq = makeSeqListConst<uint16_t>(argIter, dict, false);
			score = ((knlm::KNLangModel<uint16_t>*)inst)->evaluateLL(&seq[0], seq.size());
		}
		else if (wsize == 4)
		{
			auto seq = makeSeqListConst<uint32_t>(argIter, dict, false);
			score = ((knlm::KNLangModel<uint32_t>*)inst)->evaluateLL(&seq[0], seq.size());
		}
		Py_DECREF(dict);
		Py_DECREF(argIter);
		return Py_BuildValue("f", score);
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* knlm__evaluateSent(PyObject* self, PyObject* args)
{
	PyObject *argSelf, *argIter, *item;
	float minValue = -100;
	if (!PyArg_ParseTuple(args, "OO|f", &argSelf, &argIter, &minValue)) return nullptr;
	try
	{
		PyObject* instObj = PyObject_GetAttrString(argSelf, "_inst");
		if (!instObj) throw runtime_error{ "_inst is null" };
		PyObject* wsizeObj = PyObject_GetAttrString(argSelf, "_wsize");
		knlm::IModel* inst = (knlm::IModel*)PyLong_AsLongLong(instObj);
		size_t wsize = PyLong_AsLong(wsizeObj);
		Py_DECREF(instObj);
		Py_DECREF(wsizeObj);

		if (!(argIter = PyObject_GetIter(argIter)))
		{
			throw runtime_error{ "argIter is not iterable" };
		}

		PyObject* dict = PyObject_GetAttrString(argSelf, "_dict");
		float score = 0;
		if (wsize == 1)
		{
			auto seq = makeSeqListConst<uint8_t>(argIter, dict);
			score = ((knlm::KNLangModel<uint8_t>*)inst)->evaluateLLSent(&seq[0], seq.size(), minValue);
		}
		else if (wsize == 2)
		{
			auto seq = makeSeqListConst<uint16_t>(argIter, dict);
			score = ((knlm::KNLangModel<uint16_t>*)inst)->evaluateLLSent(&seq[0], seq.size(), minValue);
		}
		else if (wsize == 4)
		{
			auto seq = makeSeqListConst<uint32_t>(argIter, dict);
			score = ((knlm::KNLangModel<uint32_t>*)inst)->evaluateLLSent(&seq[0], seq.size(), minValue);
		}
		Py_DECREF(dict);
		Py_DECREF(argIter);
		return Py_BuildValue("f", score);
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* knlm__evaluateEachWord(PyObject* self, PyObject* args)
{
	PyObject *argSelf, *argIter, *item;
	float minValue = -INFINITY;
	if (!PyArg_ParseTuple(args, "OO|f", &argSelf, &argIter, &minValue)) return nullptr;
	try
	{
		PyObject* instObj = PyObject_GetAttrString(argSelf, "_inst");
		if (!instObj) throw runtime_error{ "_inst is null" };
		PyObject* wsizeObj = PyObject_GetAttrString(argSelf, "_wsize");
		knlm::IModel* inst = (knlm::IModel*)PyLong_AsLongLong(instObj);
		size_t wsize = PyLong_AsLong(wsizeObj);
		Py_DECREF(instObj);
		Py_DECREF(wsizeObj); 

		if (!(argIter = PyObject_GetIter(argIter)))
		{
			throw runtime_error{ "argIter is not iterable" };
		}

		PyObject* dict = PyObject_GetAttrString(argSelf, "_dict");
		vector<float> scores;
		if (wsize == 1)
		{
			auto seq = makeSeqListConst<uint8_t>(argIter, dict, false);
			scores = ((knlm::KNLangModel<uint8_t>*)inst)->evaluateLLEachWord(&seq[0], seq.size());
		}
		else if (wsize == 2)
		{
			auto seq = makeSeqListConst<uint16_t>(argIter, dict, false);
			scores = ((knlm::KNLangModel<uint16_t>*)inst)->evaluateLLEachWord(&seq[0], seq.size());
		}
		else if (wsize == 4)
		{
			auto seq = makeSeqListConst<uint32_t>(argIter, dict, false);
			scores = ((knlm::KNLangModel<uint32_t>*)inst)->evaluateLLEachWord(&seq[0], seq.size());
		}
		Py_DECREF(dict);
		Py_DECREF(argIter);
		PyObject* ret = PyList_New(scores.size() - 1);
		for (size_t i = 1; i < scores.size(); ++i)
		{
			PyList_SetItem(ret, i - 1, Py_BuildValue("f", max(scores[i], minValue)));
		}
		return ret;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* knlm__branchingEntropy(PyObject* self, PyObject* args)
{
	PyObject *argSelf, *argIter, *item;
	if (!PyArg_ParseTuple(args, "OO", &argSelf, &argIter)) return nullptr;
	try
	{
		PyObject* instObj = PyObject_GetAttrString(argSelf, "_inst");
		if (!instObj) throw runtime_error{ "_inst is null" };
		PyObject* wsizeObj = PyObject_GetAttrString(argSelf, "_wsize");
		knlm::IModel* inst = (knlm::IModel*)PyLong_AsLongLong(instObj);
		size_t wsize = PyLong_AsLong(wsizeObj);
		Py_DECREF(instObj);
		Py_DECREF(wsizeObj);

		if (!(argIter = PyObject_GetIter(argIter)))
		{
			throw runtime_error{ "argIter is not iterable" };
		}

		PyObject* dict = PyObject_GetAttrString(argSelf, "_dict");
		float score = 0;
		if (wsize == 1)
		{
			auto seq = makeSeqListConst<uint8_t>(argIter, dict, false);
			score = ((knlm::KNLangModel<uint8_t>*)inst)->branchingEntropy(&seq[0], seq.size());
		}
		else if (wsize == 2)
		{
			auto seq = makeSeqListConst<uint16_t>(argIter, dict, false);
			score = ((knlm::KNLangModel<uint16_t>*)inst)->branchingEntropy(&seq[0], seq.size());
		}
		else if (wsize == 4)
		{
			auto seq = makeSeqListConst<uint32_t>(argIter, dict, false);
			score = ((knlm::KNLangModel<uint32_t>*)inst)->branchingEntropy(&seq[0], seq.size());
		}
		Py_DECREF(dict);
		Py_DECREF(argIter);
		return Py_BuildValue("f", score);
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* knlm__save(PyObject* self, PyObject* args)
{
	PyObject *argSelf;
	const char* path;
	if (!PyArg_ParseTuple(args, "Os", &argSelf, &path)) return nullptr;
	try
	{
		PyObject* instObj = PyObject_GetAttrString(argSelf, "_inst");
		if (!instObj) throw runtime_error{ "_inst is null" };
		PyObject* wsizeObj = PyObject_GetAttrString(argSelf, "_wsize");
		knlm::IModel* inst = (knlm::IModel*)PyLong_AsLongLong(instObj);
		size_t wsize = PyLong_AsLong(wsizeObj);
		Py_DECREF(instObj);
		Py_DECREF(wsizeObj);
		inst->writeToStream(ofstream{ path + string{".mdl"}, ios_base::binary });

		PyObject *pickle = PyImport_ImportModule("pickle"), *io = PyImport_ImportModule("io");
		PyObject *file = PyObject_CallMethod(io, "open", "ss", (path + string{".dict"}).c_str(), "wb");
		if (!file)
		{
			Py_XDECREF(io);
			Py_XDECREF(pickle);
			return nullptr;
		}
		PyObject* dict = PyObject_GetAttrString(argSelf, "_dict");
		PyObject_CallMethod(pickle, "dump", "OO", dict, file);
		Py_DECREF(dict);
		Py_XDECREF(file);
		Py_XDECREF(io);
		Py_XDECREF(pickle);

		Py_INCREF(Py_None);
		return Py_None;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}


static PyObject* knlm__load(PyObject* self, PyObject* args)
{
	const char* path;
	if (!PyArg_ParseTuple(args, "s", &path)) return nullptr;
	try
	{
		PyObject* newInst = PyObject_CallFunction(gClass, nullptr);
		if (!newInst) return nullptr;
		PyObject* instObj = PyObject_GetAttrString(newInst, "_inst");
		if (!instObj) throw runtime_error{ "_inst is null" };
		PyObject* wsizeObj = PyObject_GetAttrString(newInst, "_wsize");
		knlm::IModel* inst = (knlm::IModel*)PyLong_AsLongLong(instObj);
		size_t wsize = PyLong_AsLong(wsizeObj);
		Py_DECREF(instObj);
		Py_DECREF(wsizeObj);
		delete inst;
		do
		{
			inst = new knlm::KNLangModel<uint8_t>;
			try
			{
				inst->readFromStream(ifstream{ path + string{".mdl"}, ios_base::binary });
				wsize = 1;
				break;
			}
			catch (const runtime_error& e)
			{
				delete inst;
			}

			inst = new knlm::KNLangModel<uint16_t>;
			try
			{
				inst->readFromStream(ifstream{ path + string{".mdl"}, ios_base::binary });
				wsize = 2;
				break;
			}
			catch (const runtime_error& e)
			{
				delete inst;
			}

			inst = new knlm::KNLangModel<uint32_t>;
			inst->readFromStream(ifstream{ path + string{".mdl"}, ios_base::binary });
			wsize = 4;
		} while (0);

		PyObject_SetAttrString(newInst, "_inst", PyLong_FromLongLong((ssize_t)inst));
		PyObject_SetAttrString(newInst, "_wsize", Py_BuildValue("n", wsize));
		PyObject *pickle = PyImport_ImportModule("pickle"), *io = PyImport_ImportModule("io");
		PyObject *file = PyObject_CallMethod(io, "open", "ss", (path + string{ ".dict" }).c_str(), "rb");
		if (!file)
		{
			Py_XDECREF(io);
			Py_XDECREF(pickle);
			return nullptr;
		}
		PyObject_SetAttrString(newInst, "_dict",  PyObject_CallMethod(pickle, "load", "O", file));
		Py_XDECREF(file);
		Py_XDECREF(io);
		Py_XDECREF(pickle);
		
		return newInst;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* knlm__getattr(PyObject* self, PyObject* args)
{
	PyObject *argSelf;
	const char* name;
	if (!PyArg_ParseTuple(args, "Os", &argSelf, &name)) return nullptr;
	try
	{
		PyObject* instObj = PyObject_GetAttrString(argSelf, "_inst");
		if (!instObj) throw runtime_error{ "_inst is null" };
		PyObject* wsizeObj = PyObject_GetAttrString(argSelf, "_wsize");
		knlm::IModel* inst = (knlm::IModel*)PyLong_AsLongLong(instObj);
		size_t wsize = PyLong_AsLong(wsizeObj);
		Py_DECREF(instObj);
		Py_DECREF(wsizeObj);
		if (name == string("order"))
		{
			return Py_BuildValue("n", inst->getOrder());
		}
		else if (name == string("vocabs"))
		{
			return Py_BuildValue("n", inst->getVocabSize());
		}
		else
		{
			return PyErr_Format(PyExc_AttributeError, "%s", name);
		}
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject *createClassObject(const char *name, PyMethodDef methods[])
{
	PyObject *pClassName = PyUnicode_FromString(name);
	PyObject *pClassBases = PyTuple_New(0);
	PyObject *pClassDic = PyDict_New();


	PyMethodDef *def;
	for (def = methods; def->ml_name != NULL; def++)
	{
		PyObject *func = PyCFunction_New(def, NULL);
		PyObject *method = PyInstanceMethod_New(func);
		PyDict_SetItemString(pClassDic, def->ml_name, method);
		Py_DECREF(func);
		Py_DECREF(method);
	}

	PyObject *pClass = PyObject_CallFunctionObjArgs((PyObject *)&PyType_Type, pClassName, pClassBases, pClassDic, NULL);

	Py_DECREF(pClassName);
	Py_DECREF(pClassBases);
	Py_DECREF(pClassDic);

	return pClass;
}


PyMODINIT_FUNC PyInit_knlm_c()
{
	static PyMethodDef methods[] =
	{
		{ nullptr, nullptr, 0, nullptr }
	};
	static PyModuleDef mod =
	{
		PyModuleDef_HEAD_INIT,
		"knlm_c",
		"Modified Kneser-Ney Language Model Module for Python",
		-1,
		methods
	};
	static PyMethodDef clsMethods[] =
	{
		{ "__init__", knlm__init, METH_VARARGS, "initializer" },
		{ "train", knlm__train, METH_VARARGS, "train a sequence" },
		{ "optimize", knlm__optimize, METH_VARARGS, "optimize" },
		{ "evaluate", knlm__evaluate , METH_VARARGS, "evaluate ll of last element" },
		{ "evaluateSent", knlm__evaluateSent, METH_VARARGS, "evaluate total ll of sequences" },
		{ "evaluateEachWord", knlm__evaluateEachWord, METH_VARARGS, "evaluate each sequence" },
		{ "branchingEntropy", knlm__branchingEntropy, METH_VARARGS, "evaluate branching entropy of sequence" },
		{ "__getattr__", knlm__getattr, METH_VARARGS, "getattr" },
		{ "save", knlm__save, METH_VARARGS, "save current trained model to file" },
		{ "load", knlm__load, METH_VARARGS | METH_STATIC, "load model from file" },
		{ "__del__", knlm__del, METH_VARARGS, "destructor" },
		{ nullptr, nullptr, 0, nullptr }
	};
	gModule = PyModule_Create(&mod);
	PyObject *pModuleDic = PyModule_GetDict(gModule);
	PyDict_SetItemString(pModuleDic, "KneserNey", gClass = createClassObject("KneserNey", clsMethods));
	if (!PyEval_ThreadsInitialized()) {
		PyEval_InitThreads();
	}
	return gModule;
}
